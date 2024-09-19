import pathlib
from pathlib import Path
from os import listdir
from os.path import isfile, join
from rdkit import Chem

import docking_utils

HERE = pathlib.Path().resolve()
PATH_HYDE = Path("/Applications/Hydescorer.app/Contents/MacOS/Hydescorer")
PATH_FLEXX = Path("/Applications/FlexX.app/Contents/MacOS/FlexX")
DOCKING_CONFIG = HERE / "data" / "docking_config" / "5n1f"
com_path = Path(HERE / "data" / "recombination_library")
docking_input = com_path / "temp"


all_files = [f for f in listdir(docking_input) if isfile(join(docking_input, f))]

ligand_files = []
template_files = []
mols_docking = [
    mol for mol in Chem.SDMolSupplier(str(com_path / "combinatorial_library.sdf"))
]

print(f"Read {len(mols_docking)} molecules from file")

for file in all_files:
    if "template" in file:
        template_files.append(file)
    else:
        ligand_files.append(file)

template_files = sorted(template_files)
ligand_files = sorted(ligand_files)
print("Template files:")
print(template_files)
print("Ligand files:")
print(ligand_files)

concat_docked = docking_utils.full_docking(
    ligand_files,
    template_files,
    com_path / "temp" / "docking_output",
    DOCKING_CONFIG,
    PATH_FLEXX,
    PATH_HYDE,
    10,
)

print(f"done docking, concat_docking len: {len(concat_docked)}")
final_docked = docking_utils.filter_poses_id(mols_docking, concat_docked)
print(
    f"Successful dockings: {len([f for f in final_docked if f is not None])} / {len(mols_docking)}"
)
