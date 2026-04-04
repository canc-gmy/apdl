from analysis.Plate import Plate
from analysis.OptimizedPlate import OptimizedPlate
from analysis.OptimizedPlateDistributed import OptimizedPlateDistributed
from pathlib import Path
import os
from time import time_ns
import numpy as np



cwd = Path(os.getcwd())

csv = cwd / "dist"
csv.mkdir(exist_ok=True)

# Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -10.0               # Concentrated load [N]
a = 1                # X length of plate [m]
b = 1              # Y length of plate [m]
t = 0.002              # Thickness [m]
            # Y position of load [m]

space_res_x = 0.002   # Spatial resolution X direction of results [m]
space_res_y = 0.002   # Spatial resolution Y direction of results [m]

plate = OptimizedPlateDistributed(a, b, t, space_res_x, space_res_y, 0, 0, 1, 1, D, P, 5000)

# for xi in np.arange(0.006, 0.1, 0.006):
#     for eta in np.arange(0.006, 0.1, 0.006):
#         t0 = time_ns()
#         plate.xi = xi
#         plate.eta = eta
#         plate.dataframe_centroid_with_eps.to_parquet(csv / f"ana_{xi:.3f}_{eta:.3f}.parquet")
#         t1 = time_ns()
#         print((t1 - t0) / 1e9)

xi = 0.5
eta = 0.5


t0 = time_ns()
plate.xi = xi
plate.eta = eta
plate.dataframe_centroid_with_eps.to_parquet(csv / f"ana_{xi:.3f}_{eta:.3f}.parquet")
t1 = time_ns()
print((t1 - t0) / 1e9)


