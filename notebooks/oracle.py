import docking_utils
import utils
import frag_modifications
import acquisition_filtering
from pathlib import Path
from rdkit import Chem
import useful_rdkit_utils as uru

import pandas as pd
import random
import sys
from collections import Counter

sys.path.insert(
    1,
    "/Users/paulakramer/Documents/PHD/KinFragLib/kinase_active_learning/notebooks/recombination",
)
from recombination import recombine_ligands
from recombination import cli


# A parent is given up on after this many fruitless modification attempts. It follows the
# generation budget instead of being a parameter of its own: the budget is what a cycle may
# produce, and the extra 100 is slack for modifications that come back empty.
ATTEMPT_HEADROOM = 100


class Oracle:
    def __init__(self, df, subset_df, affinity_file):
        self.subpockets = ["AP", "FP", "GA", "SE"]
        self.df = utils.get_subset_library(df, subset_df, self.subpockets)
        self.affinities = docking_utils.read_affinities(str(affinity_file))
        self.path = Path.cwd()

    def get_affinities(self, mols):
        lig_names = [
            mol.GetProp("frag_ids") if mol is not None else None for mol in mols
        ]
        lig_names = [
            utils.sort_ligand_name(name) if name is not None else None
            for name in lig_names
        ]
        final_affinities = []
        for lig in lig_names:
            if lig is None:
                final_affinities.append(None)
            else:
                try:
                    final_affinities.append(self.affinities[lig][0])
                except KeyError:
                    final_affinities.append(None)
        # final_affinities = [self.affinities[lig][0] if lig is not None else None for lig in lig_names]
        index = [ind for ind, aff in enumerate(final_affinities) if aff is not None]
        return final_affinities, index

    def get_all_affinities(self):
        return self.affinities

    def sample_one_pocket(self, num, pocket):
        sample_df = self.df[pocket].sample(num)
        return (
            sample_df.fp.values,
            self.get_values(sample_df.ROMol_dummy),
            sample_df.smiles,
        )

    def sample(self, num):
        pocket_sample = random.choices(self.subpockets, k=num)
        counter = Counter(pocket_sample).most_common()
        sample_df = {}
        for count in counter:
            sample_df[count[0]] = self.df[count[0]].sample(count[1])
            for ind in sample_df[count[0]].index:
                sample_df[count[0]].loc[ind].ROMol_original.SetProp("Index", str(ind))
        lib_path = self.path / ".." / "data" / "temp_library"
        utils.delete_folder(lib_path)

        utils.save_fragment_library_to_sdfs(lib_path, pd.concat(sample_df))
        com_path = Path(self.path / ".." / "data" / "temp" / "recombined")
        cli.process(com_path, lib_path, "AP", 3)
        # writes recombined ligands to combinatorial_library.sdf
        recombine_ligands.process(self.df, com_path)
        mols = utils.read_sdf(com_path / "combinatorial_library_large.sdf")
        # remap mol props from temp_lib to kinfraglib
        return (
            list(map(uru.mol2numpy_fp, mols)),
            [Chem.MolToSmiles(m) for m in mols],
            mols,
        )

    @staticmethod
    def parent_quotas(n_parents, max_generated, minimum=1):
        """New molecules each parent may contribute, an equal share each.

        The budget is shared out so that the first parent cannot use up the whole cycle
        and leave the others with nothing to do. `minimum` keeps the parents from
        dropping out entirely when there are more of them than the budget allows.
        """
        if n_parents <= 0:
            return []
        return [max(minimum, round(max_generated / n_parents))] * n_parents

    def fragment_sampling(
        self, num, mols, vals, failure_frag_ids, clustering, max_generated=1000
    ):
        # num: number of top scoring molecules used as parents for fragment modifications
        # max_generated: max. new molecules per cycle, shared out evenly over the parents
        max_attempts = max_generated + ATTEMPT_HEADROOM
        if clustering:
            # perform diversity picking
            zipped = zip(mols, vals)
            zipped = sorted(zipped, key=lambda x: x[1], reverse=True)[:50]
            mols, vals = zip(*zipped)
            mols = list(mols)
            vals = list(vals)
            indexes_filtered = acquisition_filtering.compound_filtering(
                mols, vals, num, []
            )
            zipped = list(
                zip(
                    utils.indexing(mols, indexes_filtered),
                    utils.indexing(vals, indexes_filtered),
                )
            )
        else:
            # greedy approach
            zipped = zip(mols, vals)
            zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
        parents = zipped[: min(num, len(zipped))]
        quotas = self.parent_quotas(len(parents), max_generated)
        ligands = []
        res_tmp = []
        for (mol, _), quota in zip(parents, quotas):
            frag_ids, bond_ids = utils.extract_meta(mol)
            # this parent's share, still capped by the budget for the whole cycle
            target = min(max_generated, len(res_tmp) + quota)
            count = 0
            # query fragment exchange multiple times
            # print("new ligand")
            while (len(res_tmp) < target) and (count < max_attempts):
                # print("count: {count}")
                count += 1
                res = frag_modifications.fragment_modification(
                    frag_ids, bond_ids, self.df, self.subpockets
                )
                if res is None:
                    continue
                if res[0] is None:
                    continue
                if (
                    utils.sort_ligand_name(res[0].GetProp("frag_ids")) not in res_tmp
                ) and (
                    utils.sort_ligand_name(res[0].GetProp("frag_ids"))
                    not in failure_frag_ids
                ):
                    res_tmp.append(utils.sort_ligand_name(res[0].GetProp("frag_ids")))
                    ligands.append(res[0])
        return (
            list(map(uru.mol2numpy_fp, ligands)),
            [Chem.MolToSmiles(m) for m in ligands],
            ligands,
        )

    def random_sampling(self, num, mols, vals, max_generated=1000):
        # same as fragment_sampling, but the parents are chosen randomly
        max_attempts = max_generated + ATTEMPT_HEADROOM
        zipped = list(zip(mols, vals))
        random.shuffle(zipped)
        parents = zipped[: min(num, len(zipped))]
        quotas = self.parent_quotas(len(parents), max_generated)
        ligands = []
        res_tmp = []
        for (mol, _), quota in zip(parents, quotas):
            frag_ids, bond_ids = utils.extract_meta(mol)
            # this parent's share, still capped by the budget for the whole cycle
            target = min(max_generated, len(res_tmp) + quota)
            count = 0
            # query fragment exchange multiple times
            while (len(res_tmp) < target) and (count < max_attempts):
                # print("count: {count}")
                count += 1
                res = frag_modifications.fragment_modification(
                    frag_ids, bond_ids, self.df, self.subpockets
                )
                if res is None:
                    continue
                if res[0] is None:
                    continue
                if Chem.MolToSmiles(res[0]) not in res_tmp:
                    res_tmp.append(Chem.MolToSmiles(res[0]))
                    ligands.append(res[0])
        return (
            list(map(uru.mol2numpy_fp, ligands)),
            [Chem.MolToSmiles(m) for m in ligands],
            ligands,
        )
