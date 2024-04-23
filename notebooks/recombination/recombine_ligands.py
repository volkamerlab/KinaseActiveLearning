#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.PropertyMol import PropertyMol

from pickle_loader import pickle_loader
from functools import reduce
import pandas as pd

import multiprocessing as mp
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def read_fragment_library(path_to_library, subpockets):
    """
    Read fragment library

    Parameters
    ----------
    path_to_library: PosixPath
        path to the folder containing the fragment library

    Returns
    -------
    data: dict(list(RDKit Molecule))
        dictionary with list of fragments for each subpocket

    """

    logger.info(f"Read fragment library...")
    logger.info(f"Path: {path_to_library}")

    data = {}
    for subpocket in subpockets:

        file = path_to_library / (subpocket + ".sdf")

        # read molecules
        # keep hydrogen atoms
        suppl = Chem.SDMolSupplier(str(file), removeHs=False)
        mols = [f for f in suppl]

        fragments = []
        for i, fragment in enumerate(mols):

            fragment = Chem.RemoveHs(fragment)

            # store unique atom identifiers
            for a, atom in enumerate(fragment.GetAtoms()):
                frag_atom_id = f"{subpocket}_{a}"
                atom.SetProp("frag_atom_id", frag_atom_id)
                atom.SetProp("frag_id", fragment.GetProp("complex_pdb"))

            fragment = PropertyMol(fragment)

            fragments.append(fragment)

        data[subpocket] = fragments

        n_frags = len(fragments)
        logger.info(f"Number of fragments in {subpocket}: {n_frags}")

    return data


def standardize_mol(mol):
    """
    Standardize molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule.

    Returns
    -------
    rdkit.Chem.rdchem.Mol or None
        Standardized molecule or None if standardization failed.
    """

    try:

        # sanitize molecule
        Chem.SanitizeMol(mol)

        # remove non-explicit hydrogens
        mol = Chem.RemoveHs(mol)

        # disconnect metals from molecule
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)

        # normalize moleucle
        mol = rdMolStandardize.Normalize(mol)

        # reionize molecule
        mol = rdMolStandardize.Reionize(mol)

        # uncharge molecule (this helps to standardize protonation states)
        u = rdMolStandardize.Uncharger()
        mol = u.uncharge(mol)

        # assign stereochemistry
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        return mol

    except Exception as e:

        logger.info(f"ERROR in standardization: {e}")
        return None


def recombine_ligand(meta, data):
    """
    Construct a ligand by connecting multiple fragments based on a Combination object

    Parameters
    ----------
    meta: Combination object
        Molecule to be constructed
    data: dict(Mol)
        dictionary containing a list of fragments for each subpocket

    Returns
    -------
    ligand: RDKit Molecule or None
        None if the ligand was not constructed
        RDKit Molecule else.

    """

    frag_ids = meta.frag_ids
    bonds = [tuple(bond) for bond in meta.bonds]

    fragments = []
    for frag_id in frag_ids:
        subpocket = frag_id[:2]
        idx = int(frag_id[3:])
        fragment = data[subpocket][idx]
        fragments.append(fragment)

    # combine fragments using map reduce model
    combo = reduce(Chem.CombineMols, fragments)

    bonds_matching = True
    ed_combo = Chem.EditableMol(combo)
    replaced_dummies = []
    for bond in bonds:

        dummy_1 = next(
            atom for atom in combo.GetAtoms() if atom.GetProp("frag_atom_id") == bond[0]
        )
        dummy_2 = next(
            atom for atom in combo.GetAtoms() if atom.GetProp("frag_atom_id") == bond[1]
        )
        atom_1 = dummy_1.GetNeighbors()[0]
        atom_2 = dummy_2.GetNeighbors()[0]

        # check bond types
        bond_type_1 = combo.GetBondBetweenAtoms(
            dummy_1.GetIdx(), atom_1.GetIdx()
        ).GetBondType()
        bond_type_2 = combo.GetBondBetweenAtoms(
            dummy_2.GetIdx(), atom_2.GetIdx()
        ).GetBondType()
        if bond_type_1 != bond_type_2:
            bonds_matching = False
            break

        ed_combo.AddBond(atom_1.GetIdx(), atom_2.GetIdx(), order=bond_type_1)

        replaced_dummies.extend([dummy_1.GetIdx(), dummy_2.GetIdx()])

    # remove replaced dummy atoms
    replaced_dummies.sort(reverse=True)
    for dummy in replaced_dummies:
        ed_combo.RemoveAtom(dummy)

    ligand = ed_combo.GetMol()

    # replace remaining dummy atoms with hydrogens
    du = Chem.MolFromSmiles("*")
    h = Chem.MolFromSmiles("[H]", sanitize=False)
    ligand = AllChem.ReplaceSubstructs(ligand, du, h, replaceAll=True)[0]
    try:
        ligand = Chem.RemoveHs(ligand)
    except ValueError:
        logger.error(Chem.MolToSmiles(ligand))
        return

    # do not construct this ligand if bond types are not matching
    if not bonds_matching:
        return

    # clear properties
    for prop in ligand.GetPropNames():
        ligand.ClearProp(prop)
    for atom in ligand.GetAtoms():
        atom.ClearProp("frag_atom_id")

    return ligand


def bond_to_str(bonds):
    bond_str = ""
    for b in bonds:
        bond_str += ",".join(b)
        bond_str += "|"
    return bond_str[:-1]


def ligand_recombination(meta, fragment_library):
    # construct ligand
    try:
        ligand = recombine_ligand(meta, fragment_library)
        ligand.SetProp("frag_ids", ",".join(meta.frag_ids))
        ligand.SetProp("bond_ids", bond_to_str(meta.bonds))
    except Exception as e:
        print(
            f"Error {e}: Molecule construction failed for molecule with fragment IDs {meta.frag_ids}"
        )
        return None

    # standardize molecule
    try:
        ligand = standardize_mol(ligand)
    except Exception as e:
        print(
            f"Error {e}: Molecule standardization failed for molecule with fragment IDs {meta.frag_ids}"
        )
        return None

    return ligand


def combinatory_library(path_combinatorial_library, fragment_library):
    # n_cores = mp.cpu_count()
    # logger.info(f"Number of cores: {n_cores}")
    # pool = mp.Pool(n_cores)

    results = []

    # get all paths to pickle files with molecules from recombination
    paths_pickle_combinatorial_library = list(
        (path_combinatorial_library / "results").glob("*.pickle")
    )

    logger.info(f"Process recombined ligands...")

    # iterate over pickle files
    for i, path_pickle_combinatorial_library in enumerate(
        paths_pickle_combinatorial_library
    ):
        logger.info(f"Process {path_pickle_combinatorial_library}...")

        with open(str(path_pickle_combinatorial_library), "rb") as pickle_in:
            # process ligands in pickle file (returns list of dict)
            results_tmp = [
                ligand_recombination(meta, fragment_library)
                for meta in pickle_loader(pickle_in)
            ]
            # for meta in pickle_loader(pickle_in):
            #    temp = ligand_recombination(meta, fragment_library)
            # args_pool = [(meta, fragment_library) for meta in pickle_loader(pickle_in)]
            # results_tmp = pool.starmap(
            #    ligand_recombination,
            #    args_pool,
            # )

            logger.info(f"Number of ligands in current iteration: {len(results_tmp)}")

            # extend results list with ligands from current iteration
            results.extend(results_tmp)
    results = [r for r in results if r is not None]
    logger.info(f"Number of recombined ligands from all iterations: {len(results)}")
    print(f"Number of recombined ligands from all iterations: {len(results)}")
    # logger.info(f'Data linked to each ligand: {list(results[0].keys())}')

    with Chem.SDWriter(
        str(path_combinatorial_library / "combinatorial_library.sdf")
    ) as f:
        for m in results:
            if m is not None:
                f.write(m)


def main():
    subpockets = ["AP", "FP", "SE", "GA"]
    HERE = Path().parent.resolve()
    fragment_library = read_fragment_library(
        HERE / ".." / "data" / "temp_library", subpockets
    )

    combinatory_library(
        HERE / ".." / "data" / "temp" / "recombined",
        fragment_library,
    )


if __name__ == "__main__":
    main()
