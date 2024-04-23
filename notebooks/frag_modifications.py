import random
from functools import reduce


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PropertyMol import PropertyMol


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


def get_terminal_fragments(bond_id, subpockets):
    count = {s: 0 for s in subpockets}
    for bond in bond_id:
        count[bond[0][:2]] += 1
        count[bond[1][:2]] += 1
    return [k for k in count.keys() if count[k] == 1]


def remove_bond(bond_id, subpocket):
    return [b for b in bond_id if subpocket not in b[0] and subpocket not in b[1]]


def remove_fragid(fragment_id, subpocket):
    return [f for f in fragment_id if subpocket not in f]


def fragment_del(fragment_id, bond_id, fragment_library):
    subpockets = get_terminal_fragments(bond_id, fragment_library.keys())
    frag_del = random.choice(subpockets)
    del_frag_id = remove_fragid(fragment_id, frag_del)
    del_bond_id = remove_bond(bond_id, frag_del)
    del_ligand = construct_ligand(del_frag_id, del_bond_id, fragment_library)
    return del_ligand, del_frag_id, del_bond_id


def fragment_add(fragment_id, bond_id, fragment_library):
    current_pockets = [s[:2] for s in fragment_id]
    add_pocket = []
    # get all possible additions
    for frag in fragment_id:
        atom_subpockets = fragment_library[frag[:2]].atom_subpockets[int(frag[3:])]
        frag_pockets = list(set(atom_subpockets.split(" ")))
        add_pocket += [
            (frag[:2], a, frag[3:]) for a in frag_pockets if a not in current_pockets
        ]

    # select position to add fragments
    # frag_opt: (subpocket of present fragment, new subpocket present fragment could connect to, present fragment id)
    frag_opt = random.choice(add_pocket)

    # sample fragment to add
    frag_index = fragment_library[frag_opt[1]].sample().index[0]
    # check subpocket compatibility
    frag_pockets = list(
        set(fragment_library[frag_opt[1]].atom_subpockets[frag_index].split(" "))
    )
    if frag_opt[0] not in frag_pockets:
        # TODO return fragment_add(fragment_id, bond_id, fragment_library)
        print("hello")
        return

    # add new fragment_id
    fragment_id.append(frag_opt[1] + "_" + str(frag_index))
    # get new dummy atoms to connect
    atom_id = get_dummy_id(
        fragment_library[frag_opt[0]].iloc[int(frag_opt[2])], frag_opt[1]
    )
    atom_id1 = get_dummy_id(fragment_library[frag_opt[1]].iloc[frag_index], frag_opt[0])
    # add new bond_id
    bond_id.append(
        (frag_opt[0] + "_" + str(atom_id), frag_opt[1] + "_" + str(atom_id1))
    )
    # create molecule
    ligand = construct_ligand(fragment_id, bond_id, fragment_library)
    return ligand, fragment_id, bond_id


def get_dummy_id(fragment, subpocket):
    fragment = Chem.RemoveHs(fragment.ROMol_original)
    for i, a in enumerate(fragment.GetAtoms()):
        if a.GetProp("subpocket") == subpocket:
            assert a.GetAtomicNum() == 0
            return i
    return None


def fragment_exchange(fragment_id, bond_id, fragment_library):
    subpockets = get_terminal_fragments(bond_id, fragment_library.keys())
    exch_pocket = random.choice(subpockets)
    # save current bond id and pocket (which is to be exchanged)
    for b in bond_id:
        if exch_pocket in b[0]:
            bond_id_1 = b[1]
            neigh_pocket = b[1][:2]
        elif exch_pocket in b[1]:
            bond_id_1 = b[0]
            neigh_pocket = b[0][:2]

    # remove fragment
    del_frag_id = remove_fragid(fragment_id, exch_pocket)
    del_bond_id = remove_bond(bond_id, exch_pocket)

    # add fragment in same space
    frag_index = fragment_library[exch_pocket].sample().index[0]
    frag_pockets = list(
        set(fragment_library[exch_pocket].atom_subpockets[frag_index].split(" "))
    )
    if neigh_pocket not in frag_pockets:
        # select different fragment
        return fragment_exchange(fragment_id, bond_id, fragment_library)
    del_frag_id.append(exch_pocket + "_" + str(frag_index))
    # get dummy atom to connect from new fragment
    atom_id = get_dummy_id(fragment_library[exch_pocket].iloc[frag_index], neigh_pocket)
    del_bond_id.append((bond_id_1, exch_pocket + "_" + str(atom_id)))
    ligand = construct_ligand(del_frag_id, del_bond_id, fragment_library)
    return ligand, del_frag_id, del_bond_id


def fragment_modification(fragment_id, bond_id, fragment_library):
    # if only one fragment present, add fragment instead
    if len(fragment_id) == 1:
        return fragment_add(fragment_id, bond_id, fragment_library)

    # else: choose randomly between three options: addition, deletion or substitution
    frag_opt = random.choice(range(3))
    # fragment deletion
    if frag_opt == 0:
        return fragment_del(fragment_id, bond_id, fragment_library)
    # fragment addition
    elif frag_opt == 1:
        try:
            ligand, frag_id, bonds = fragment_add(
                fragment_id, bond_id, fragment_library
            )
        except IndexError:
            # no possibility to add, choose between delete and exchange instead
            new_opt = random.choice(range(2))
            if new_opt == 0:
                ligand, frag_id, bonds = fragment_del(
                    fragment_id, bond_id, fragment_library
                )
            elif new_opt == 1:
                ligand, frag_id, bonds = fragment_exchange(
                    fragment_id, bond_id, fragment_library
                )
        return ligand, frag_id, bonds
    # fragment exchange
    elif frag_opt == 2:
        fragment_exchange(fragment_id, bond_id, fragment_library)


def sample_close_fragments(smi, vals, n_instances, fragment_library):
    zipped = zip(smi, vals)
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    for s, _ in zipped[:n_instances]:
        # TODO: get frag_ids and bond_ids
        fragment_modification()
