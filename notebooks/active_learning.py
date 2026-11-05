"""
Active learning loop for the recombined kinase ligand space, written once for all strategies.

A strategy is described by a `Strategy` and differs in only two places:

- `query_strategy`: acquisition function from `acquisition_filtering` that picks which of the
  generated molecules are sent to the oracle in each cycle
- `parent_selection`: which already scored molecules are modified to generate new ones,
  "top scoring" (`oracle.fragment_sampling`) or "random" (`oracle.random_sampling`);
  with "top scoring", `clustering=True` picks diverse parents instead of the plain top ones

Everything else (optimizer, normalization, bookkeeping) is shared.
"""

import inspect
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
import useful_rdkit_utils as uru
from modAL.models import BayesianOptimizer
from rdkit import Chem
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm.auto import tqdm

import docking_utils
import utils
from run_utils import run_parameters, set_seed


# ---------------------------------------------------------------- model


def calculate_similarity(a, b):
    """Tanimoto similarity a vs. b."""
    aa = np.sum(a, axis=1, keepdims=True)
    bb = np.sum(b, axis=1, keepdims=True)
    ab = np.matmul(a, b.T)
    return np.true_divide(ab, aa + bb.T - ab)


class TanimotoKernel(
    gp.kernels.NormalizedKernelMixin, gp.kernels.StationaryKernelMixin, gp.kernels.Kernel
):
    def __init__(self):
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        assert not eval_gradient
        if Y is None:
            Y = X
        return calculate_similarity(X, Y)


# ---------------------------------------------------------------- settings


@dataclass
class Settings:
    """Settings shared by all strategies of one run."""

    n_initial: int = 50  # molecules in the starting set
    n_cycles: int = 20  # active learning cycles
    n_parents: int = 20  # molecules modified to generate new ones, per cycle
    n_query: int = 50  # molecules that should come back with a value, per cycle
    # a cycle keeps sending molecules to the oracle until `n_query` of them return a
    # value, but never sends more than this many in total. None spends exactly `n_query`
    # calls, so cycles that hit molecules the oracle cannot score end up short.
    max_oracle_calls: int = None
    max_generated: int = 1000  # max. new molecules generated per cycle

    def as_params(self):
        return dict(
            n_initial=self.n_initial,
            n_cycles=self.n_cycles,
            n_parents_per_cycle=self.n_parents,
            n_query_per_cycle=self.n_query,
            max_oracle_calls_per_cycle=self.max_oracle_calls,
            max_generated_per_cycle=self.max_generated,
        )


@dataclass
class Strategy:
    """One active learning strategy to run."""

    name: str  # used in output file names, e.g. "diverse"
    query_strategy: Callable  # acquisition function from acquisition_filtering
    parent_selection: str = "top scoring"  # or "random"
    clustering: bool = False  # only used with "top scoring"

    def __post_init__(self):
        allowed = ("top scoring", "random")
        if self.parent_selection not in allowed:
            raise ValueError(f"parent_selection must be one of {allowed}")

    def as_params(self):
        return dict(
            strategy=self.name,
            query_function=self.query_strategy.__name__,
            parent_selection=self.parent_selection,
            clustering=self.clustering,
        )


@dataclass
class Result:
    """Everything one strategy run produces."""

    strategy: Strategy
    params: dict
    val_list: list  # normalized oracle values, starting set first
    used: list  # SMILES in the same order as val_list
    used_mols: list  # molecules in the same order as val_list
    loop_indices: list  # how many molecules were added per cycle
    oracle_calls: list  # molecules sent to the oracle per cycle, the real cost of a run
    explored_mols: list  # all generated molecules, not only the selected ones
    failure_frag_ids: list  # fragment ids the oracle has no value for
    optimizer: object = field(default=None, repr=False)

    def results_df(self, data_mean, data_std):
        return create_results_df(
            self.loop_indices, self.used_mols, self.used, data_mean, data_std,
            self.val_list, self.oracle_calls
        )


# ---------------------------------------------------------------- run


def initial_distribution(oracle, path_mols, n_initial, seed):
    """
    Starting set of `n_initial` molecules that the oracle has a value for.

    A plain random sample, so it follows the composition of the library.
    The mean and standard deviation of this set are used to normalize all later values,
    so they are returned as well.
    """
    set_seed(seed)
    mols = utils.read_sdf(path_mols)
    # keep only molecules the oracle knows
    # the oracle stores its keys with the fragments sorted, so sort before looking up
    known = oracle.get_all_affinities()
    mols = [mol for mol in mols if utils.sort_ligand_name(mol.GetProp("frag_ids")) in known]
    print(f"{len(mols)} molecules with an oracle value")
    random.shuffle(mols)
    mols = mols[:n_initial]

    fps = list(map(uru.mol2numpy_fp, mols))
    y = oracle.get_affinities(mols)[0]
    data_mean = float(np.mean(y))
    data_std = float(np.std(y))
    y_norm = utils.normalize(y, data_mean, data_std)
    smiles = [Chem.MolToSmiles(m) for m in mols if m is not None]
    return y_norm, mols, fps, smiles, data_mean, data_std


def _sample_parents(oracle, strategy, settings, used_mols, val_list, failure_frag_ids):
    """Generates new molecules by modifying fragments of already scored molecules."""
    if strategy.parent_selection == "random":
        return oracle.random_sampling(
            settings.n_parents,
            used_mols,
            val_list,
            settings.max_generated,
        )
    return oracle.fragment_sampling(
        settings.n_parents,
        used_mols,
        val_list,
        failure_frag_ids,
        strategy.clustering,
        settings.max_generated,
    )


def _query(optimizer, query_strategy, fps, n_query, used_idx, sample_mols):
    """Asks the optimizer which molecules to score next.

    Acquisition functions differ in what they need: `sample_diverse` takes the molecules
    themselves, the others only the fingerprints. Each argument is passed only to
    functions that accept it.
    """
    accepted = inspect.signature(query_strategy).parameters
    kwargs = {"n_instances": n_query, "used": used_idx}
    if "mols" in accepted:
        kwargs["mols"] = sample_mols
    return optimizer.query(fps, **kwargs)


def run_strategy(
    oracle,
    strategy,
    settings,
    initial,
    data_mean,
    data_std,
    seed,
    run_timestamp,
    extra_params=None,
    verbose=False,
):
    """
    Runs the active learning loop for one strategy.

    Parameters
    ----------
    oracle : Oracle
    strategy : Strategy
    settings : Settings
    initial : tuple
        (values, molecules, fingerprints, SMILES) from `initial_distribution`
    data_mean, data_std : float
        normalization constants from `initial_distribution`
    seed : int
    run_timestamp : str
        from `run_utils.new_run_timestamp()`, shared by all output files of one run
    extra_params : dict, optional
        further settings to record in the .json file, e.g. input file paths
    verbose : bool
        print the oracle values of each cycle

    Returns
    -------
    Result
    """
    vals_init, mols_init, fps_init, smi_init = initial

    optimizer = BayesianOptimizer(
        estimator=GaussianProcessRegressor(kernel=TanimotoKernel()),
        X_training=np.stack(fps_init),
        y_training=vals_init,
        query_strategy=strategy.query_strategy,
    )

    val_list = list(vals_init).copy()
    used = list(smi_init).copy()
    used_mols = list(mols_init).copy()
    explored_mols = list(mols_init).copy()
    loop_indices = [len(used)]
    oracle_calls = [0]  # the starting set costs no queries of its own
    failure_frag_ids = []

    params = run_parameters(
        seed,
        run_timestamp,
        strategy.query_strategy,
        strategy.parent_selection,
        clustering=strategy.clustering,
        normalization_mean=data_mean,
        normalization_std=data_std,
        **settings.as_params(),
        **(extra_params or {}),
    )
    params["strategy"] = strategy.name

    # seed here so every strategy starts from the same state
    set_seed(seed)
    for _ in tqdm(range(settings.n_cycles), desc=strategy.name):
        # generate new molecules around molecules we already know
        fps, sample_smi, sample_mols = _sample_parents(
            oracle, strategy, settings, used_mols, val_list, failure_frag_ids
        )
        explored_mols += sample_mols

        # pick the ones to send to the oracle, skipping molecules already scored.
        # molecules the oracle cannot score are replaced, up to `max_oracle_calls`,
        # so a cycle reaches `n_query` values instead of silently coming up short.
        used_idx = list(utils.check_used(sample_smi, used))
        call_budget = settings.max_oracle_calls or settings.n_query
        cycle_vals, cycle_smi, cycle_mols, cycle_desc = [], [], [], []
        calls = 0
        while len(cycle_vals) < settings.n_query and calls < call_budget:
            want = min(settings.n_query - len(cycle_vals), call_budget - calls)
            query_ind, query_desc = _query(
                optimizer,
                strategy.query_strategy,
                fps,
                want,
                used_idx,
                sample_mols,
            )
            if not len(query_ind):
                break  # nothing left to pick from
            # never offer the same molecule twice within a cycle
            used_idx += list(query_ind)
            calls += len(query_ind)

            query_mols = utils.indexing(sample_mols, query_ind)
            query_smi = utils.indexing(sample_smi, query_ind)

            # ask the oracle; molecules without a value are remembered as failures
            vals, docking_ind = oracle.get_affinities(query_mols)
            if verbose:
                print(vals)
            failure_frag_ids += docking_utils.extract_docking_failures(query_mols, vals)

            vals = utils.indexing(vals, docking_ind)
            cycle_vals += list(np.reshape(utils.normalize(vals, data_mean, data_std), (-1)))
            cycle_smi += utils.indexing(query_smi, docking_ind)
            cycle_mols += utils.indexing(query_mols, docking_ind)
            cycle_desc += utils.indexing(query_desc, docking_ind)

        vals = cycle_vals
        query_smi, query_mols, query_desc = cycle_smi, cycle_mols, cycle_desc
        docking_ind = range(len(vals))
        oracle_calls.append(calls)

        val_list += vals
        used += query_smi
        used_mols += query_mols
        loop_indices.append(len(vals))

        if vals:
            optimizer.teach(query_desc, vals)

    return Result(
        strategy=strategy,
        params=params,
        val_list=val_list,
        used=used,
        used_mols=used_mols,
        loop_indices=loop_indices,
        oracle_calls=oracle_calls,
        explored_mols=explored_mols,
        failure_frag_ids=failure_frag_ids,
        optimizer=optimizer,
    )


def run_strategies(oracle, strategies, settings, initial, data_mean, data_std, seed,
                   run_timestamp, extra_params=None, verbose=False):
    """Runs every strategy and returns {strategy name: Result}."""
    results = {}
    for strategy in strategies:
        results[strategy.name] = run_strategy(
            oracle, strategy, settings, initial, data_mean, data_std,
            seed, run_timestamp, extra_params, verbose,
        )
    return results


# ---------------------------------------------------------------- results


def create_results_df(loop_indices, used_mols, used_smiles, data_mean, data_std, val_list,
                      oracle_calls=None):
    """One row per scored molecule, with the cycle it was selected in and its oracle value.

    `oracle_calls` is how many molecules each cycle sent to the oracle, including the ones
    that came back without a value. Its running total is the real cost of a run, so it is
    the fair x-axis when runs spend different numbers of calls per cycle.
    """
    res_list = []
    cur_ind = 0
    spent = 0
    for i, v in enumerate(loop_indices):
        spent += oracle_calls[i] if oracle_calls else v
        # undo the normalization
        res_list += [(i, spent, x * data_std + data_mean)
                     for x in val_list[cur_ind : cur_ind + v]]
        cur_ind += v
    df = pd.DataFrame(res_list, columns=["cycle", "oracle calls", "binding affinity"])
    df["SMILES"] = used_smiles
    df["Mols"] = used_mols
    df["frag_ids"] = [utils.sort_ligand_name(mol.GetProp("frag_ids")) for mol in used_mols]
    df = df.sort_values("binding affinity", ascending=False)
    df = df.drop_duplicates("SMILES").sort_index()
    return df
