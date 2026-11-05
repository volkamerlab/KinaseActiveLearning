"""
Analysis of active learning results, written once and applied to every strategy.

Each function takes one results DataFrame (from `active_learning.create_results_df`)
and returns plain lists, so they can be mapped over {strategy name: DataFrame}.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
import useful_rdkit_utils as uru

import utils

SUBPOCKETS = ["AP", "FP", "GA", "SE"]


def sort_affinities(affinities):
    """All oracle values as [(frag_ids, value)], best first."""
    pairs = [(name, value[0]) for name, value in affinities.items()]
    return sorted(pairs, key=lambda x: x[1], reverse=True)


def apply_to_results(results_dfs, function, *args, **kwargs):
    """Runs `function(df, ...)` for every strategy and returns {strategy name: result}."""
    return {name: function(df, *args, **kwargs) for name, df in results_dfs.items()}


def merge_results(results_dfs, column="strategy"):
    """One DataFrame of all strategies with a column naming the strategy, for seaborn."""
    return pd.concat([df.assign(**{column: name}) for name, df in results_dfs.items()])


# ---------------------------------------------------------------- top-k recovery


def top_fractions_per_cycle(df, sorted_affs, top_nums, n_cycles):
    """
    Fraction of the molecules selected in each cycle that are among the overall top k.

    Returns one list per entry of `top_nums`.
    """
    counts_per_cycle = df["cycle"].value_counts()
    per_top = []
    for top in top_nums:
        top_frag_ids = set(frag_ids for frag_ids, _ in sorted_affs[:top])
        hits = [0] * (n_cycles + 1)
        for _, row in df.iterrows():
            if row.frag_ids in top_frag_ids:
                hits[row.cycle] += 1
        # cycles without selected molecules count as 0 instead of raising
        per_top.append(
            [hit / counts_per_cycle.get(i, 0) if counts_per_cycle.get(i, 0) else 0.0
             for i, hit in enumerate(hits)]
        )
    return per_top


def cumulative_top_fractions(df, per_top, top_nums):
    """
    Share of the overall top k found up to and including each cycle.

    Returns (cumulative lists per top k, total share found per top k).
    """
    counts_per_cycle = df["cycle"].value_counts()
    cumulative, totals = [], []
    for top, fractions in zip(top_nums, per_top):
        per_cycle = [f * counts_per_cycle.get(i, 0) for i, f in enumerate(fractions)]
        running = np.cumsum(per_cycle)
        cumulative.append([value / top for value in running])
        totals.append(sum(per_cycle) / top)
    return cumulative, totals


def print_top_recovery(totals, top_nums, name=""):
    print(f"Share of the overall top k found{f' ({name})' if name else ''}:")
    print("Top k \t | Share found")
    print("---------|-------------")
    for top, total in zip(top_nums, totals):
        print(f"Top {top} \t | {round(total, 3)}")


# ---------------------------------------------------------------- diversity


def molecular_diversity(df, n_cycles):
    """1 - mean pairwise Tanimoto similarity of the molecules selected in each cycle."""
    if "Fingerprints" not in df.columns:
        df = add_fingerprints(df)
    diversities = []
    for i in range(n_cycles + 1):
        cycle_fp = list(df[df["cycle"] == i]["Fingerprints"])
        if not cycle_fp:
            diversities.append(np.nan)
            continue
        means = [np.mean(DataStructs.BulkTanimotoSimilarity(fp, cycle_fp)) for fp in cycle_fp]
        diversities.append(1.0 - np.mean(means))
    return diversities


def add_fingerprints(df):
    """Adds a Morgan fingerprint column, needed for molecular_diversity."""
    df = df.copy()
    mols = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
    df["Fingerprints"] = [uru.mol2morgan_fp(mol) for mol in mols]
    return df


def count_total_fragments(affinities):
    """Number of distinct fragments in the whole scored space."""
    fragments = [frag for name in affinities for frag in name.split(",")]
    return len(set(fragments))


def count_total_fragments_per_subpocket(affinities):
    """Number of distinct fragments per subpocket, in the order of SUBPOCKETS."""
    per_pocket = {pocket: set() for pocket in SUBPOCKETS}
    for name in affinities:
        for frag in utils.sort_ligand_name(name).split(","):
            pocket = frag[:2]
            if pocket in per_pocket:
                per_pocket[pocket].add(frag)
            else:
                print(f"Fragment doesn't match a subpocket: {frag}")
    return [len(per_pocket[pocket]) for pocket in SUBPOCKETS]


def fragment_diversity(df, total_fragments, n_cycles):
    """Share of all fragments seen up to and including each cycle."""
    diversity = []
    for i in range(n_cycles + 1):
        fragments = _fragments_up_to(df, i)
        diversity.append(len(fragments) / total_fragments)
    return diversity


def fragment_diversity_per_subpocket(df, total_per_subpocket, n_cycles):
    """
    Same as fragment_diversity, per subpocket.

    Returns (one list per subpocket, fragments seen per subpocket at the last cycle).
    """
    diversity = [[] for _ in SUBPOCKETS]
    seen_last = [[] for _ in SUBPOCKETS]
    for i in range(n_cycles + 1):
        fragments = _fragments_up_to(df, i)
        for j, pocket in enumerate(SUBPOCKETS):
            in_pocket = {frag for frag in fragments if frag.startswith(pocket)}
            diversity[j].append(len(in_pocket) / total_per_subpocket[j])
            seen_last[j] = sorted(in_pocket)
    return diversity, seen_last


def _fragments_up_to(df, cycle):
    frag_lists = list(df[df.cycle <= cycle].frag_ids)
    return {frag for frag_ids in frag_lists for frag in frag_ids.split(",")}


# ---------------------------------------------------------------- molecule properties


def mean_property_per_cycle(df, property_function, n_cycles):
    """
    Mean of a molecule property per cycle, e.g. QED or synthetic accessibility.

    property_function : callable taking an RDKit molecule, e.g. `QED.default`
    """
    means = []
    for i in range(n_cycles + 1):
        mols = [Chem.MolFromSmiles(smi) for smi in df[df["cycle"] == i]["SMILES"]]
        scores = [property_function(mol) for mol in mols if mol is not None]
        means.append(np.mean(scores) if scores else np.nan)
    return means


def qed_per_cycle(df, n_cycles):
    return mean_property_per_cycle(df, QED.default, n_cycles)


# ---------------------------------------------------------------- top fragments


def top_fragments_per_subpocket(sorted_affs, fraction=None, n_top=None):
    """
    How often each fragment occurs in the best part of the scored space.

    Give either `fraction` (e.g. 0.01 for the best 1%) or `n_top` (e.g. 100 for the
    best 100 ligands). Returns one {fragment: count} per subpocket, in the order
    of SUBPOCKETS.
    """
    if (fraction is None) == (n_top is None):
        raise ValueError("give either fraction or n_top")
    index = n_top if n_top is not None else int(len(sorted_affs) * fraction)
    counts = [{} for _ in SUBPOCKETS]
    for frag_ids, _ in sorted_affs[:index]:
        for frag in frag_ids.split(","):
            pocket = frag[:2]
            if pocket not in SUBPOCKETS:
                print(f"Fragment doesn't match a subpocket: {frag}")
                continue
            per_pocket = counts[SUBPOCKETS.index(pocket)]
            per_pocket[frag] = per_pocket.get(frag, 0) + 1
    return counts


def fragment_coverage(top_fragments, selected_fragments, subpocket="AP"):
    """
    Share of the top fragments of one subpocket that were seen during exploration.

    top_fragments : list of {fragment: count} per subpocket, from top_fragments_per_subpocket
    selected_fragments : list of fragment lists per subpocket, from fragment_diversity_per_subpocket
    """
    j = SUBPOCKETS.index(subpocket)
    top = set(top_fragments[j])
    found = top & set(selected_fragments[j])
    return len(found) / len(top) if top else 0.0, len(found)
