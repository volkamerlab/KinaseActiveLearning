from pathlib import Path
from rdkit import Chem
import subprocess
import os
from rdkit.Chem import AllChem
import utils
from rdkit.Chem import rdMolAlign
import subprocess
import numpy as np
from scipy.stats import halfnorm


def core_docking(
    path_fragment: Path,
    path_config: Path,
    path_output: Path,
    path_flexx: Path,
    ligand,
    print_output=False,
    num_poses=10,
    threads=10,
) -> list:
    """
    runs FlexX docking

    Returns
    ----------
    List of all poses (docking result) as Mol (Properties: docking-score, pose)

    Parameters
    ----------
    path_fragment: pathlib.path
        Path to core-fragment sdf-file
    path_config: pathlib.path
        Path to FlexX-config file.
    path_flexx: pathlib.path
        Path to FlexX
    path_output: pathlib.path
        Path to output file
    ligand: Ligand
        Ligand object of molecule to dock
    num_poses: int
        Number of poses to generate during docking, default=10
    """

    if os.path.exists(path_output):
        os.remove(path_output)
    # core docking
    output_text = subprocess.run(
        [
            str(Path(".") / path_flexx),
            "-i",
            str(path_fragment),
            "--docking-definition",
            str(path_config),
            "-o",
            str(path_output),
            "--max-nof-con",
            str(num_poses),
            "--allowed-6ring-confs",
            str(2),
            "--thread-count",
            str(threads),
        ],
        capture_output=True,
    )
    if print_output:
        print(output_text.stderr)

    # read core fragments poses from sdf
    docked_core_fragments = []
    if os.stat(
        str(path_output)
    ).st_size:  # only acces reult file if at least one pose was generated
        for molecule in Chem.SDMolSupplier(str(path_output)):
            # molecule.SetProp("frag_ids", str(ligand.GetProp("frag_ids")))
            # molecule.SetProp("bond_ids", str(ligand.GetProp("bond_ids")))
            docked_core_fragments.append(molecule)

    return docked_core_fragments


def hyde_scoring(
    path_docking_results: Path,
    path_config: Path,
    path_output: Path,
    path_hyde: Path,
    ligand,
    threads=10,
    print_output=False,
) -> list:
    """
    runs hydescoring

    Returns
    ----------
    List of all poses (docking result) as Mol (Properties: BIOSOLVEIT.DOCKING_SCORE, pose)

    Parameters
    ----------
    path_fragment: pathlib.path
        Path to fragment sdf-file
    path_config: pathlib.path
        Path to hyde-config file.
    path_output: pathlib.path
        Path to output file
    path_hyde: pathlib.path
        Path to Hyde
    ligand: Ligand
        Ligand object of molecule to score
    """

    if os.path.exists(path_output):
        os.remove(path_output)
    output_text = subprocess.run(
        [
            str(Path(".") / path_hyde),
            "-i",
            str(path_docking_results),
            "--binding-site-definition",
            str(path_config),
            "-o",
            str(path_output),
            "--thread-count",
            str(threads),
        ],
        capture_output=True,
    )
    if print_output:
        print(output_text.stderr)

    # read results from sdf
    opt_fragments = []
    if (
        os.stat(str(path_output)).st_size and os.stat(str(path_docking_results)).st_size
    ):  # only access result file if at least one pose was generated
        for molecule_docking, molecule_opt in zip(
            Chem.SDMolSupplier(str(path_docking_results)),
            Chem.SDMolSupplier(str(path_output)),
        ):
            # clear all hyde properties, that are not needed
            superfluos_props = [
                "BIOSOLVEIT.HYDE_ATOM_SCORES [kJ/mol]",
                "BIOSOLVEIT.HYDE_LIGAND_EFFICIENCY range: ++, +, 0, -, --",
                "BIOSOLVEIT.HYDE_LIGAND_LIPOPHILIC_EFFICIENCY range: ++, +, 0, -, --",
                "BIOSOLVEIT.INTER_CLASH range: red, yellow, green",
                "BIOSOLVEIT.INTRA_CLASH range: red, yellow, green",
                "BIOSOLVEIT.INTRA_CLASH range: red, yellow, green",
                "BIOSOLVEIT.MOLECULE_CHECKSUM",
                "BIOSOLVEIT.TORSION_QUALITY range: red, yellow, green, not rotatable",
            ]
            for prop in superfluos_props:
                molecule_opt.ClearProp(prop)
            try:
                # set proporties
                molecule_opt.SetProp("frag_ids", str(ligand.GetProp("frag_ids")))
                molecule_opt.SetProp("bond_ids", str(ligand.GetProp("bond_ids")))
            except:
                # core fragment doesn't have frag_id and bond_id
                pass
            # copy docking from docked molecule to optimzed molecule
            molecule_opt.SetProp(
                "BIOSOLVEIT.DOCKING_SCORE",
                molecule_docking.GetProp("BIOSOLVEIT.DOCKING_SCORE"),
            )
            opt_fragments.append(molecule_opt)
    return opt_fragments


def compute_coordinates(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    # the molecule now has a crude conformation, clean it up:
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def template_docking(
    path_fragment: Path,
    path_template: Path,
    path_config: Path,
    path_output: Path,
    path_flexx: Path,
    ligand,
    print_output=False,
    num_poses=10,
    threads=10,
) -> list:
    """
    runs FlexX template-docking

    Returns
    ----------
    List of all poses (docking result) as Mol (Properties: BIOSOLVEIT.DOCKING_SCORE, pose)

    Parameters
    ----------
    path_fragment: pathlib.path
        Path to fragment sdf-file
    path_template: pathlib.path
        Path to template-fragment sdf-file
    path_config: pathlib.path
        Path to FlexX-config file.
    path_flexx: pathlib.path
        Path to Flexx
    path_output: pathlib.path
        Path to output file
    ligand: Ligand
        Ligand of molecule to dock
    num_poses: int
        Number of poses to generate during docking, default=10
    """
    # template docking
    output_text = subprocess.run(
        [
            str(path_flexx),
            "-i",
            str(path_fragment),
            "--docking-definition",
            str(path_config),
            "-o",
            str(path_output),
            "-t",
            str(path_template),
            "--docking-type",
            "1",
            # "--stereo-mode"
            # ,"3"
            "--max-nof-con",
            str(num_poses),
            "--allowed-6ring-confs",
            str(2),
            "--thread-count",
            str(threads),
        ],
        capture_output=True,  # needed to capture output text
    )
    if print_output:
        print(output_text.stderr)

    # read fragments poses from sdf
    docked_fragments = []
    if os.stat(str(path_output)).st_size:
        for molecule in Chem.SDMolSupplier(str(path_output)):
            # molecule.SetProp("frag_ids", str(ligand.GetProp("frag_ids")))
            # molecule.SetProp("bond_ids", str(ligand.GetProp("bond_ids")))
            docked_fragments.append(molecule)

    return docked_fragments


def extract_fragment(mol, subpocket, fragment_library, path_fragment):
    frag_id = ""
    for frag in mol.GetProp("frag_ids").split(","):
        if subpocket in frag:
            ind = int(frag[3:])
            frag_id = frag
            break

    fragment = fragment_library[subpocket].iloc[ind].ROMol
    fragment.SetProp("frag_ids", frag_id)
    fragment = compute_coordinates(fragment)
    utils.write_to_sdf([fragment], path_fragment)
    return fragment


def docking(
    path_ligand: Path,
    path_config: Path,
    path_output: Path,
    path_flexx: Path,
    path_hyde: Path,
    fragment_library: list,
    ligand,
    print_output=False,
    threads=10,
) -> list:

    path_template = path_output / "AP_fragment.sdf"
    # extract AP fragment
    fragment = extract_fragment(ligand, "AP", fragment_library, path_template)

    # AP docking + hyde
    docked = core_docking(
        path_template,
        path_config / "AP.flexx",
        path_output / "core_docking_output.sdf",
        path_flexx,
        fragment,
        print_output,
        threads,
    )

    hyde_docked = hyde_scoring(
        path_output / "core_docking_output.sdf",
        path_config / "AP.hydescorer",
        path_output / "core_hyde_output.sdf",
        path_hyde,
        fragment,
        print_output,
        threads,
    )

    # chosen_poses = choose_template_poses(path_output / "core_hyde_output.sdf")

    chosen_poses = utils.read_sdf(path_output / "core_hyde_output.sdf")
    docked = []
    for pose in chosen_poses:
        utils.write_to_sdf([pose], path_output / "core_docking_pose.sdf")
        # template docking with AP fragment as template
        template_docked = template_docking(
            path_ligand,
            path_output / "core_docking_pose.sdf",
            path_config / "subpocket.flexx",
            path_output / "template_docking.sdf",
            path_flexx,
            ligand,
            print_output,
            threads,
        )

        hyde_template = hyde_scoring(
            path_output / "template_docking.sdf",
            path_config / "subpocket.hydescorer",
            path_output / "template_hyde.sdf",
            path_hyde,
            ligand,
            print_output,
            threads,
        )
        docked.append(hyde_template[0])

    return hyde_template


def choose_template_poses(path_poses, num_templates=3):
    """
    Chooses *num_templates*-best poses according to docking result and diversity and removes the non-top-scored or invalid poses

    Parameters
    ----------
    num_templates: int
        max. templates to choose, if None, num_templates == len(path)
    """
    # self.poses_pre_filtered = self.poses.copy()

    poses = utils.read_sdf(path_poses)
    if not len(poses):
        # nothing to choose from
        return
    if not num_templates:
        # default of num_poses = amount of poses
        return

    # always choose the pose with the best docking score
    chosen_poses = [
        min(
            poses,
            key=lambda p: float(
                p.GetProp("BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]")
            ),
        )
    ]
    poses.remove(chosen_poses[0])

    # define mapping of atoms => avoid a maximal common substructure search (not needed because the structures should be the same)
    atom_mapping = [[j, j] for j in range(chosen_poses[0].GetNumAtoms())]

    # append num_templates - 1 more templates according to highest mean of RMSD (to already choosen poses) and the lowest docking-score
    for _ in range(num_templates - 1):
        if not len(poses):
            break
        # choose minimal pose according to -2 * mean(RMSD to choosen poses) + dockin_score
        pose = min(
            poses,
            key=lambda pose: -2
            * sum(rdMolAlign.CalcRMS(pose, x, map=[atom_mapping]) for x in chosen_poses)
            / len(chosen_poses)
            + float(
                pose.GetProp("BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]")
            ),
        )
        chosen_poses.append(pose)
        poses.remove(pose)

    return chosen_poses


def calculate_rescoring(mols, path_input, path_output):
    utils.write_to_sdf(mols, path_input)
    subprocess.call(
        f"./../docking_rescoring/rf-score-vs --receptor ../docking_rescoring/5n1f_seesar.pdb {path_input} -O {path_output}",
        shell=True,
    )
    mols = utils.read_sdf(path_output)
    affinity = [float(m.GetProp("RFScoreVS_v2")) for m in mols]
    return mols, affinity


def extract_pocket(mol, pocket):
    frag_ids = mol.GetProp("frag_ids").split(",")
    for frag in frag_ids:
        if pocket in frag:
            return frag
    print(mol.GetProp("frag_ids"))
    return None
    # frag_ids = [(p.split("_")[0], p.split("_")[1]) for p in frag_ids]
    # pockets, ind = frag_ids.unzip()
    # try:
    #    p_ind = pockets.index(pocket)
    #    return
    # except ValueError:
    #    print(f"Pocket not covered in mol: {pocket}, {mol.GetProp("frag_ids")}")
    #    return None


# def remove_dummies(mol):
# replace dummy atoms with hydrogens
# du = Chem.MolFromSmiles("*")
# h = Chem.MolFromSmiles("[H]", sanitize=False)
# ligand = AllChem.ReplaceSubstructs(mol, du, h, replaceAll=True)[0]
# try:
#    ligand = Chem.RemoveHs(ligand)
#    return ligand
# except ValueError as ve:
#    print(ve)
#    return


def rename_ligands(lig_file):
    mols = [mol for mol in Chem.SDMolSupplier(lig_file)]
    for mol in mols:
        mol.SetProp("_Name", mol.GetProp("frag_ids"))
    utils.write_to_sdf(mols, lig_file)


def create_template_docking_files(mols, core_frags, out_file, frag_lib):
    mols = np.array(mols)
    core_uniqes = list(set(core_frags))
    core_mols = [utils.get_fragment_mol(core, frag_lib) for core in core_uniqes]
    core_frags = np.array(core_frags)
    ligand_file_names = []
    template_file_names = []
    # iterate over all possible core fragments
    for i, core in enumerate(core_uniqes):
        ind = np.where(core_frags == core)[0]
        ligands = mols[ind]
        core_file = str(out_file) + "_" + core + ".sdf"
        ligand_file_names.append(core_file)
        # write all molecules with same core fragment into one file
        utils.write_to_sdf(ligands, core_file)
        template_file = str(out_file) + "_template_" + core + ".sdf"
        template_file_names.append(template_file)
        # write matching core fragment (without dummy atoms) into single file
        core_coords = compute_coordinates(core_mols[i])
        utils.write_to_sdf([core_coords], template_file)
    return ligand_file_names, template_file_names


def full_docking(
    ligand_files,
    template_files,
    output_path,
    docking_config,
    path_flexx,
    path_hyde,
    threads=10,
):
    concat_docked = []
    for lig_file, templ_file in zip(ligand_files, template_files):
        # AP fragment docking
        core_frag = [mol for mol in Chem.SDMolSupplier(templ_file, removeHs=False)]
        docking = core_docking(
            templ_file,
            docking_config / "AP.flexx",
            output_path / "mol_core_docked.sdf",
            path_flexx,
            core_frag[0],
            True,
            10,
            threads,
        )
        if len(docking) == 0:
            continue
        hyde_docked = hyde_scoring(
            output_path / "mol_core_docked.sdf",
            docking_config / "AP.hydescorer",
            output_path / "mol_core_hyde.sdf",
            path_hyde,
            core_frag[0],
            True,
            threads,
        )
        if len(hyde_docked) == 0:
            continue
        # template docking
        core_poses = [
            mol for mol in Chem.SDMolSupplier(str(output_path / "mol_core_hyde.sdf"))
        ]
        core_poses = [
            mol
            for mol in Chem.SDMolSupplier(
                str(output_path / "mol_core_docked.sdf"), removeHs=False
            )
        ]
        rename_ligands(lig_file)
        mols = [mol for mol in Chem.SDMolSupplier(lig_file, removeHs=False)]
        for i, core_pose in enumerate(core_poses):
            file_name = "mol_core_hyde_pose_" + str(i) + ".sdf"
            templ_file_name = "mol_template_docking_pose_" + str(i) + ".sdf"
            final_file_name = "mol_template_hyde_pose_" + str(i) + ".sdf"
            utils.write_to_sdf([core_pose], output_path / file_name)
            template_docked = template_docking(
                lig_file,
                output_path / file_name,
                docking_config / "AP_FP_GA_SE_clashes.flexx",
                output_path / templ_file_name,
                path_flexx,
                mols,
                True,
            )
            print(f"Number of dockings after template docking: {len(template_docked)}")
            if len(template_docked) == 0:
                continue
            template_hyde = hyde_scoring(
                output_path / templ_file_name,
                docking_config / "AP_FP_GA_SE_clashes.hydescorer",
                output_path / final_file_name,
                path_hyde,
                mols,
                True,
                threads,
            )
            # rint(f"Number of dockings after hyde optimization: {len(template_hyde)}")
            concat_docked += template_hyde
            # concat_docked += template_docked
    return concat_docked


def filter_poses(mols, docked_mols):
    smi_docked = [Chem.MolToSmiles(m) for m in mols]
    print(smi_docked)
    final_docked = [None for i in range(len(mols))]
    for mol in docked_mols:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        try:
            # TODO: save best pose?
            ind = smi_docked.index(smi)
        except Exception as e:
            print(e)
            continue
        final_docked[ind] = mol
    return final_docked


def filter_poses_id(mols, docked_mols):
    frag_ids = [m.GetProp("frag_ids") for m in mols]
    final_docked = [None for i in range(len(mols))]
    for mol in docked_mols:
        if mol is None:
            continue
        frags = "_".join(mol.GetProp("_Name").split("_")[:-1])
        try:
            ind = frag_ids.index(frags)
        except Exception as e:
            print(e)
            continue
        if final_docked[ind] is None:
            final_docked[ind] = mol
        # save best docking pose only
        # if float(final_docked[ind].GetProp("BIOSOLVEIT.DOCKING_SCORE")) > float(
        #    mol.GetProp("BIOSOLVEIT.DOCKING_SCORE")
        # ):
        if float(
            final_docked[ind].GetProp(
                "BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]"
            )
        ) > float(
            mol.GetProp("BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]")
        ):
            final_docked[ind] = mol
    return final_docked


def set_default_affinity(affinities, docked_mols, default=5.5):
    indexes = [ind for ind, mol in enumerate(docked_mols) if mol is not None]
    output_aff = [default for _ in docked_mols]
    for ind, aff in zip(indexes, affinities):
        output_aff[ind] = aff
    return output_aff


def sample_decoy_value(affinities, docked_mols, mean=6.1, scale=0.2, cutoff=5.96):
    # sample decoy values from halfnorm distribution
    size = len(np.where(np.array(affinities) > 6.1)[0])
    samples = halfnorm.rvs(loc=0, scale=scale, size=size)
    samples = mean - samples
    samples = samples[np.where(samples <= cutoff)]

    # set docking affinities in array
    indexes = [ind for ind, mol in enumerate(docked_mols) if mol is not None]
    output_aff = [None for _ in docked_mols]
    for ind, aff in zip(indexes, affinities):
        output_aff[ind] = aff

    # fill docking fails with decoy values
    ind_samples = 0
    for ind, aff in enumerate(output_aff):
        if len(samples) == ind_samples:
            # some values might still be none!
            return output_aff
        if aff is None:
            output_aff[ind] = samples[ind_samples]
            ind_samples += 1
    return output_aff


# path_output = Path(
#    "/Users/paulakramer/Documents/PHD/KinFragLib/KinaseActiveLearning/data/temp/docking"
# )
# path_ligand = Path(
#    "/Users/paulakramer/Documents/PHD/KinFragLib/KinaseActiveLearning/data/temp/docking/mol_input_1.sdf"
# )
# path_config = Path(
#    "/Users/paulakramer/Documents/PHD/KinFragLib/KinaseActiveLearning/data/docking_config/5n1f/"
# )
# PATH_HYDE = Path("/Applications/Hydescorer.app/Contents/MacOS/Hydescorer")
# PATH_FLEXX = Path("/Applications/FlexX.app/Contents/MacOS/FlexX")

# PATH_TO_LIB = Path(
#    "/Users/paulakramer/Documents/PHD/KinFragLib/KinaseActiveLearning/data/fragment_library_custom_filtered"
# )

# fragment_library = utils.read_fragment_library(PATH_TO_LIB)

# mol = utils.read_sdf(path_ligand)[0]
# mol = compute_coordinates(mol)
# utils.write_to_sdf([mol], path_ligand)

# docking_results = docking(
#    path_ligand,
#    path_config,
#    path_output,
#    PATH_FLEXX,
#    PATH_HYDE,
#    fragment_library,
#    mol,
#    True,
# )
