import os
from time import time_ns
from pathlib import Path
from venv import create
from ansys.mapdl.core import Mapdl
from ansys.mapdl.core import launch_mapdl
from src.apdl import create_model_and_solve_simply_supported_edges, create_model_and_solve_simply_supported_vertices  
import numpy as np
from src.postproc import rst_to_parquet
from analysis.Plate import Plate


cwd = Path(os.getcwd())

csv = cwd / "csv"
csv.mkdir(exist_ok=True)
rst = cwd / "rst"
rst.mkdir(exist_ok=True)

# Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0               # Concentrated load [N]
a = 1                # X length of plate [m]
b = 1              # Y length of plate [m]
t = 0.002              # Thickness [m]

space_res_x = 0.002    # Spatial resolution X direction of results [m]
space_res_y = 0.002    # Spatial resolution Y direction of results [m]

mapdl = launch_mapdl()

for xi in np.arange(0.02, 0.5, 0.02):
    for eta in np.arange(0.02, 0.5, 0.02):
      print(f"solving for xi: {xi:.3f}, eta: {eta:.3f}")
      plate = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, 5000)
      plate.dataframe_centroid.to_csv(csv / f"ana_{xi:.3f}_{eta:.3f}.csv")

      create_model_and_solve_simply_supported_edges(mapdl, xi, eta)

      mapdl.db.save(rst / f"sim_{xi:.3f}_{eta:.3f}.db", "ALL")
      mapdl.post1()
      mapdl.reswrite(rst / f"sim_{xi:.3f}_{eta:.3f}")
      rst_to_parquet(rst / f"sim_{xi:.3f}_{eta:.3f}.rst", csv / f"sim_{xi:.3f}_{eta:.3f}.csv")
      mapdl.db.clear()

mapdl.exit()


