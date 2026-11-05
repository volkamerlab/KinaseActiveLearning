"""
Runs the active learning grid: every strategy on the same starting set, for several seeds.

One process per seed, so the seeds run in parallel. The defaults reproduce the current
results, the 10 seed grid with clustered parents:

    python run_grid.py --out ../Results/init100_clustered_parents_10seeds

That folder was filled in two batches, which is why its files carry two timestamps:
first `diverse` and `greedy` with clustered parents, then the `random` baseline. The
command above runs all three in one go, so its files share one timestamp.

Each strategy writes a .csv and a .json of its parameters into the output folder, named
so that `binding_affinity_stability_init100.ipynb` and `results_diversity_init100.ipynb`
can read them back. What each seed printed is kept in <out>/logs/seed<n>.log.

Existing result files are never overwritten: a rerun into a folder that already holds
them stops with an error instead, so pick a new --out or move the old files away.
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from rdkit import RDLogger

import acquisition_filtering
import active_learning as al
import oracle as ora
import utils
from run_utils import new_run_timestamp, save_results

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).parent
PATH_FRAGMENT_LIB = HERE / ".." / "data" / "fragment_library_custom_filtered"
PATH_SUBSET_LIB = HERE / ".." / "data" / "diverse_recomb_library_larger"
PATH_AFFINITIES = (
    HERE / ".." / "data" / "recombination_library"
    / "predictions_skip_hyde_random_split_large.csv"
)
PATH_INITIAL_MOLS = (
    HERE / ".." / "data" / "recombination_library" / "combinatorial_library_large.sdf"
)

STRATEGIES = [
    al.Strategy(
        "diverse",
        query_strategy=acquisition_filtering.sample_diverse,
        parent_selection="top scoring",
    ),
    al.Strategy(
        "greedy",
        query_strategy=acquisition_filtering.greedy,
        parent_selection="top scoring",
    ),
    al.Strategy(
        "random",
        query_strategy=acquisition_filtering.random_selection,
        parent_selection="random",
    ),
]

# each worker gets one thread, so the seeds running next to each other do not fight over
# the cores. This is what the saved runs used.
SINGLE_THREADED = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def parse_seeds(text):
    """"0-9", "0,1,2", "0-4,7" or "3" as a sorted list of seeds."""
    seeds = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if last < first:
                raise argparse.ArgumentTypeError(f"empty seed range {part!r}")
            seeds.update(range(first, last + 1))
        elif part.isdigit():
            seeds.add(int(part))
        else:
            raise argparse.ArgumentTypeError(f"cannot read {part!r} as a seed or range")
    if not seeds:
        raise argparse.ArgumentTypeError("no seeds given")
    return sorted(seeds)


def pick_strategies(names, clustering):
    """The strategies to run, in the order given, with clustering applied if asked for."""
    strategies = STRATEGIES
    if names:
        wanted = [n.strip() for n in names.split(",")]
        by_name = {s.name: s for s in STRATEGIES}
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown strategies {unknown}, pick from {list(by_name)}")
        strategies = [by_name[n] for n in wanted]
    if clustering:
        # the random baseline draws its parents at random, so clustering does not apply
        strategies = [
            replace(s, clustering=True) if s.parent_selection == "top scoring" else s
            for s in strategies
        ]
    return strategies


def run_seed(args, seed, run_timestamp, strategies):
    """Runs every strategy for one seed, all from the same starting set."""
    fragment_library = utils.read_fragment_library(PATH_FRAGMENT_LIB)
    subset_library = utils.read_fragment_library(PATH_SUBSET_LIB)
    oracle = ora.Oracle(fragment_library, subset_library, PATH_AFFINITIES)

    settings = al.Settings(
        n_initial=args.n_initial,
        n_cycles=args.n_cycles,
        n_parents=args.n_parents,
        n_query=args.n_query,
        max_oracle_calls=args.max_oracle_calls,
        max_generated=args.max_generated,
    )

    # one starting set per seed, shared by every strategy so the runs are comparable
    vals, mols, fps, smi, data_mean, data_std = al.initial_distribution(
        oracle, PATH_INITIAL_MOLS, settings.n_initial, seed
    )
    print(f"seed {seed}: starting set of {len(mols)} molecules, "
          f"mean {data_mean:.3f}, std {data_std:.3f}", flush=True)

    extra_params = dict(
        initial_library=PATH_INITIAL_MOLS,
        affinity_file=PATH_AFFINITIES,
        fragment_library=PATH_FRAGMENT_LIB,
        subset_library=PATH_SUBSET_LIB,
        run_mode=f"run_grid.py, one process per seed, tag {args.tag}",
    )

    for strategy in strategies:
        result = al.run_strategy(
            oracle,
            strategy,
            settings,
            (vals, mols, fps, smi),
            data_mean,
            data_std,
            seed=seed,
            run_timestamp=run_timestamp,
            extra_params=extra_params,
        )
        df = result.results_df(data_mean, data_std)
        save_results(
            df, f"active_learning_kinodata_{strategy.name}_{args.tag}", args.out, result.params
        )
        print(f"seed {seed} {strategy.name}: {len(df)} molecules, "
              f"best {df['binding affinity'].max():.3f}", flush=True)


def forwarded_arguments(args, run_timestamp, seed):
    """The command line for one worker, one seed of the grid."""
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--seeds", str(seed),
        "--out", str(args.out),
        "--timestamp", run_timestamp,
        "--n-initial", str(args.n_initial),
        "--n-cycles", str(args.n_cycles),
        "--n-parents", str(args.n_parents),
        "--n-query", str(args.n_query),
        "--max-generated", str(args.max_generated),
        "--tag", args.tag,
        "--clustering" if args.clustering else "--no-clustering",
    ]
    if args.max_oracle_calls is not None:
        command += ["--max-oracle-calls", str(args.max_oracle_calls)]
    if args.strategies:
        command += ["--strategies", args.strategies]
    return command


def run_grid(args, seeds, run_timestamp):
    """Runs every seed in its own process, `args.jobs` of them at a time."""
    log_folder = args.out / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **SINGLE_THREADED}

    def run_one(seed):
        log_path = log_folder / f"seed{seed}.log"
        with open(log_path, "w") as log:
            finished = subprocess.run(
                forwarded_arguments(args, run_timestamp, seed),
                cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
        state = "done" if finished.returncode == 0 else f"FAILED ({finished.returncode})"
        print(f"seed {seed}: {state}, log in {log_path}", flush=True)
        return seed, finished.returncode

    print(f"{len(seeds)} seeds, {args.jobs} at a time, timestamp {run_timestamp}",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(pool.map(run_one, seeds))

    failed = [seed for seed, code in outcomes if code != 0]
    if failed:
        raise SystemExit(f"seeds that failed: {failed} (see {log_folder})")
    print(f"all {len(seeds)} seeds done, results in {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", "--seed", default="0-9", type=parse_seeds,
                        help="seeds to run, e.g. 0-9, 0,1,2 or 4 (default: 0-9)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--jobs", type=int, default=None,
                        help="seeds to run at a time, each in its own single threaded "
                             "process (default: half the cores)")
    parser.add_argument("--timestamp", default=None, help="shared by all seeds of one grid")
    parser.add_argument("--n-initial", type=int, default=100)
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--n-parents", type=int, default=20,
                        help="scored molecules modified to generate new ones, per cycle")
    parser.add_argument("--n-query", type=int, default=50)
    parser.add_argument("--max-generated", type=int, default=1000,
                        help="new molecules generated per cycle")
    parser.add_argument("--clustering", action=argparse.BooleanOptionalAction, default=True,
                        help="pick the parents spread over AP clusters instead of taking "
                             "the plain top scorers (only affects 'top scoring' "
                             "strategies, so not the random baseline)")
    parser.add_argument("--strategies", default=None,
                        help="comma separated subset to run, e.g. greedy,diverse "
                             "(default: all of them)")
    parser.add_argument("--max-oracle-calls", type=int, default=None,
                        help="molecules a cycle may send to the oracle while topping up "
                             "to --n-query values (default: exactly --n-query, no top-up)")
    parser.add_argument("--tag", default="clustered_init100", help="part of the file name")
    args = parser.parse_args()

    run_timestamp = args.timestamp or new_run_timestamp()
    # the timestamp ends up in every file name, so reject a malformed one now rather
    # than after the run, when the results would be lost
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_timestamp):
        raise SystemExit(f"--timestamp must look like 2026-10-26_17-31-23, got {run_timestamp!r}")
    args.out.mkdir(parents=True, exist_ok=True)

    # fail on an unknown strategy name before anything is run
    strategies = pick_strategies(args.strategies, args.clustering)

    if len(args.seeds) == 1:
        run_seed(args, args.seeds[0], run_timestamp, strategies)
        return

    if args.jobs is None:
        args.jobs = min(len(args.seeds), max(1, (os.cpu_count() or 4) // 2))
    run_grid(args, args.seeds, run_timestamp)


if __name__ == "__main__":
    main()
