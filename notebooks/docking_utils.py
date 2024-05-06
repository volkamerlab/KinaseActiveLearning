from pathlib import Path
from rdkit import Chem
import subprocess
import os


def core_docking(
    path_fragment: Path,
    path_config: Path,
    path_output: Path,
    path_flexx: Path,
    ligand,
    print_output=False,
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
            molecule.SetProp("frag_ids", str(ligand.GetProp("frag_ids")))
            molecule.SetProp("bond_ids", str(ligand.GetProp("bond_ids")))
            docked_core_fragments.append(molecule)

    return docked_core_fragments


def hyde_scoring(
    path_docking_results: Path,
    path_config: Path,
    path_output: Path,
    path_hyde: Path,
    ligand,
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
            # set proporties
            molecule_opt.SetProp("frag_ids", str(ligand.GetProp("frag_ids")))
            molecule_opt.SetProp("bond_ids", str(ligand.GetProp("bond_ids")))
            # copy docking from docked molecule to optimzed molecule
            molecule_opt.SetProp(
                "BIOSOLVEIT.DOCKING_SCORE",
                molecule_docking.GetProp("BIOSOLVEIT.DOCKING_SCORE"),
            )
            opt_fragments.append(molecule_opt)
    return opt_fragments
