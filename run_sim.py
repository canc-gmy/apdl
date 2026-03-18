import os
from time import time_ns
from pathlib import Path
from venv import create
from ansys.mapdl.core import Mapdl
from ansys.mapdl.core import launch_mapdl 
import numpy as np
from src.postproc import extract_shell_layer_to_parquet
from src.postproc import rst_to_parquet
from src.apdl import create_model_and_solve_simply_supported_edges


cwd = Path(os.getcwd())
rst = cwd / "rst"
rst.mkdir(exist_ok=True)

csv = cwd / "sim"
csv.mkdir(exist_ok=True)


# mapdl = launch_mapdl()

# for xi in np.arange(0.006, 0.1, 0.006):
#     for eta in np.arange(0.006, 0.1, 0.006):
#         create_model_and_solve_simply_supported_edges(mapdl, xi, eta)
#         mapdl.db.save(rst / f"sim_{xi:.3f}_{eta:.3f}.db", "ALL")
#         mapdl.post1()
#         mapdl.reswrite(rst / f"sim_{xi:.3f}_{eta:.3f}")
#         extract_shell_layer_to_parquet(rst / f"sim_{xi:.3f}_{eta:.3f}.rst", csv / f"sim_{xi:.3f}_{eta:.3f}.parquet", "Top")
#         mapdl.db.clear()

# mapdl.exit()

for xi in np.arange(0.006, 0.1, 0.006):
    for eta in np.arange(0.006, 0.1, 0.006):
        extract_shell_layer_to_parquet(rst / f"sim_{xi:.3f}_{eta:.3f}.rst", csv / f"sim_{xi:.3f}_{eta:.3f}.parquet", "Top")