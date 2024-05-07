from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.MolStandardize import rdMolStandardize

import pandas as pd
from functools import reduce
from pathlib import Path
import json
import multiprocessing as mp
import logging
import os, shutil
import random

# from recombination.pickle_loader import pickle_loader
import pickle

logger = logging.getLogger(__name__)


def read_fragment_library(path_to_lib):
    """
    Read fragment library from sdf files (one file per subpocket).

    Parameters
    ----------
    path_to_lib : str
        Path to fragment library folder.


    Returns
    -------
    dict of pandas.DataFrame
        Fragment details, i.e. SMILES, kinase groups, and fragment RDKit molecules, (values) for
        each subpocket (key).
    """
    # list of folders for each subpocket
    subpockets = ["AP", "FP", "SE", "GA", "B1", "B2", "X"]

    data = {}

    # iterate over subpockets
    for subpocket in subpockets:

        try:
            data[subpocket] = _read_subpocket_fragments(subpocket, path_to_lib)
        except OSError:
            pass

    return data


def _read_subpocket_fragments(subpocket, path_to_lib):
    """
    Read fragments for input subpocket.

    Parameters
    ----------
    subpocket : str
        Subpocket name, i.e. AP, SE, FP, GA, B1, or B2.
    path_to_lib : str
        Path to fragment library folder.

    Returns
    -------
    pandas.DataFrame
        Fragment details, i.e. SMILES, kinase groups, and fragment RDKit molecules, for input
        subpocket.
    """

    mol_supplier = Chem.SDMolSupplier(
        str(path_to_lib / f"{subpocket}.sdf"), removeHs=False
    )

    data = []

    for mol in mol_supplier:

        # Replace dummy atoms with hydrogens in fragments
        dummy = Chem.MolFromSmiles("*")
        hydrogen = Chem.MolFromSmiles("[H]", sanitize=False)
        mol_wo_dummy = AllChem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)[
            0
        ]

        # Remove all hydrogens but explicit hydrogens
        mol_wo_dummy = Chem.RemoveHs(mol_wo_dummy)
        mol_w_dummy = Chem.RemoveHs(mol)

        # Generate SMILES
        smiles_wo_dummy = Chem.MolToSmiles(mol_wo_dummy)
        smiles_w_dummy = Chem.MolToSmiles(mol_w_dummy)

        # 2D coordinates
        AllChem.Compute2DCoords(mol_wo_dummy)
        AllChem.Compute2DCoords(mol_w_dummy)

        # Add property information stored for each fragment, e.g. kinase group
        data.append(
            [
                mol_wo_dummy,
                mol_w_dummy,
                mol,
                mol_w_dummy.GetProp("kinase"),
                mol_w_dummy.GetProp("family"),
                mol_w_dummy.GetProp("group"),
                mol_w_dummy.GetProp("complex_pdb"),
                mol_w_dummy.GetProp("ligand_pdb"),
                mol_w_dummy.GetProp("alt"),
                mol_w_dummy.GetProp("chain"),
                mol_w_dummy.GetProp("atom.prop.subpocket"),
                mol_w_dummy.GetProp("atom.prop.environment"),
                smiles_wo_dummy,
                smiles_w_dummy,
            ]
        )

    fragment_library = pd.DataFrame(
        data,
        columns="ROMol ROMol_dummy ROMol_original kinase family group complex_pdb ligand_pdb alt chain atom_subpockets atom_environments smiles smiles_dummy".split(),
    )
    fragment_library["subpocket"] = subpocket

    return fragment_library


def construct_ligand(fragment_ids, bond_ids, fragment_library):
    """
    Construct a ligand by connecting multiple fragments based on a Combination object

    Parameters
    ----------
    fragment_ids: list of str
        Fragment IDs of recombined ligand, e.g. `["SE_2", "AP_0", "FP_2"]` (`<subpocket>_<fragment index in subpocket pool>`).
    bond_ids : list of list of str
        Bond IDs of recombined ligand, e.g. `[["FP_6", "AP_10"], ["AP_11", "SE_13"]]`: Atom (`<subpocket>_<atom ID>`) pairs per fragment bond.
    fragment_library : dict of pandas.DataFrame
        SMILES and RDKit molecules for fragments (values) per subpocket (key).

    Returns
    -------
    ligand: rdkit.Chem.rdchem.Mol or None
        Recombined ligand (or None if the ligand could not be constructed)
    """

    fragments = []
    for fragment_id in fragment_ids:

        # Get subpocket and fragment index in subpocket
        subpocket = fragment_id[:2]
        fragment_index = int(fragment_id[3:])
        fragment = fragment_library[subpocket].ROMol_original[fragment_index]

        # Store unique atom identifiers in original molecule (important for recombined ligand construction based on atom IDs)
        fragment = Chem.RemoveHs(fragment)
        for i, atom in enumerate(fragment.GetAtoms()):
            fragment_atom_id = f"{subpocket}_{i}"
            atom.SetProp("fragment_atom_id", fragment_atom_id)
            atom.SetProp("fragment_id", fragment.GetProp("complex_pdb"))
        fragment = PropertyMol(fragment)

        # Append fragment to list of fragments
        fragments.append(fragment)

    # Combine fragments using map-reduce model
    combo = reduce(Chem.CombineMols, fragments)

    bonds_matching = True
    ed_combo = Chem.EditableMol(combo)
    replaced_dummies = []

    atoms = combo.GetAtoms()

    for bond in bond_ids:

        dummy_1 = next(
            atom
            for atom in combo.GetAtoms()
            if atom.GetProp("fragment_atom_id") == bond[0]
        )
        dummy_2 = next(
            atom
            for atom in combo.GetAtoms()
            if atom.GetProp("fragment_atom_id") == bond[1]
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

    # Do not construct this ligand if bond types are not matching
    if not bonds_matching:
        return

    # Remove replaced dummy atoms
    replaced_dummies.sort(reverse=True)
    for dummy in replaced_dummies:
        ed_combo.RemoveAtom(dummy)

    ligand = ed_combo.GetMol()

    # Replace remaining dummy atoms with hydrogens
    du = Chem.MolFromSmiles("*")
    h = Chem.MolFromSmiles("[H]", sanitize=False)
    ligand = AllChem.ReplaceSubstructs(ligand, du, h, replaceAll=True)[0]
    try:
        ligand = Chem.RemoveHs(ligand)
    except ValueError:
        print(Chem.MolToSmiles(ligand))
        return

    # Clear properties
    for prop in ligand.GetPropNames():
        ligand.ClearProp(prop)
    for atom in ligand.GetAtoms():
        atom.ClearProp("fragment_atom_id")

    # Generate 2D coordinates
    AllChem.Compute2DCoords(ligand)

    return ligand


def save_fragment_library_to_sdfs(path_output, fragment_library_concat):
    # copied from /notebooks/kinfraglib/3_1_fragment_library_reduced.ipynb
    """
    Save fragment library to file (for each subpocket sdf file).

    Parameters
    ----------
    path_output : str or pathlib.Path
        Path to output folder for sdf files.
    fragment_library_concat : pandas.DataFrame
        Fragment library data for one or multiple subpockets.
    """

    path_output = Path(path_output)
    path_output.mkdir(parents=True, exist_ok=True)

    for subpocket, fragments in fragment_library_concat.groupby("subpocket"):

        with open(path_output / f"{subpocket}.sdf", "w") as f:
            w = Chem.SDWriter(f)
            for mol in fragments.ROMol_original:
                w.write(mol)
            w.close()


def pickle_loader(pickle_file):
    """
    Load a pickle file with multiple objects

    Parameters
    ----------
    pickle_file: file object
        input binary pickle file

    Returns
    -------
    Generator object with loaded objects

    """

    try:
        while True:
            yield pickle.load(pickle_file)
    except EOFError:
        pass


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


def ligand_recombination(meta, fragment_library):
    # construct ligand
    try:
        ligand = recombine_ligand(meta, fragment_library)
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
    n_cores = mp.cpu_count()
    logger.info(f"Number of cores: {n_cores}")
    pool = mp.Pool(n_cores)

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
            results_tmp = pool.starmap(
                ligand_recombination,
                [(meta, fragment_library) for meta in pickle_loader(pickle_in)],
            )

            logger.info(f"Number of ligands in current iteration: {len(results_tmp)}")

            # extend results list with ligands from current iteration
            results.extend(results_tmp)

    results = [r for r in results if r is not None]
    logger.info(f"Number of recombined ligands from all iterations: {len(results)}")
    # logger.info(f'Data linked to each ligand: {list(results[0].keys())}')

    with Chem.SDWriter(
        str(path_combinatorial_library / "combinatorial_library.sdf")
    ) as f:
        for m in results:
            if m is not None:
                f.write(m)
    return results


def read_sdf(path):
    suppl = Chem.SDMolSupplier(str(path))
    mols = [mol for mol in suppl]
    return mols


def delete_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def check_used(smi, used):
    idx = []
    for i, s in enumerate(smi):
        if s in used:
            idx.append(i)
    return idx


def extract_meta(mol):
    frag_ids = mol.GetProp("frag_ids").split(",")
    bond_ids = mol.GetProp("bond_ids").split("|")
    bonds = []
    for b in bond_ids:
        bond = b.split(",")
        # only one fragment present -> no bonds
        if len(bond) == 1:
            return frag_ids, []
        bonds.append((bond[0], bond[1]))
    return frag_ids, bonds


def write_to_sdf(mols, path):
    with Chem.SDWriter(path) as w:
        for m in mols:
            w.write(m)


def indexing(arr, ind):
    return [arr[i] for i in ind]
