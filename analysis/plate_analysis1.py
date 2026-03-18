import numpy as np
import matplotlib.pyplot as plt
from Plate import Plate
# from analysis import Analysis # Ensure Analysis exists if you intend to run it

# Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0               # Concentrated load [N]
a = 0.5                # X length of plate [m]
b = 0.5                # Y length of plate [m]
t = 0.002              # Thickness [m]

xi = 0.01               # X position of load [m]
eta = 0.01              # Y position of load [m]

space_res_x = 0.002    # Spatial resolution X direction of results [m]
space_res_y = 0.002    # Spatial resolution Y direction of results [m]


plate = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, 8000)

# print(plate.dataframe)

plate.dataframe_centroid.to_csv(r"csv\ana_0.010_0.010.csv")

# print(len(np.arange(0.001, 0.5, 0.002)))