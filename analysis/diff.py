from os import read

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
except Exception as e:
    print(f"Could not enable LaTeX rendering: {e}\nFalling back to default.")

a = pd.read_csv(r"C:\Users\rossi\Documents\tesi\apdl\csv\plate.csv")
s = pd.read_csv(r"C:\Users\rossi\Documents\tesi\apdl\csv\test_01.csv")

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     a["X"], 
#     a["Y"], 
#     c=abs((s["EXX"]-a["EXX"])/a["EXX"]), 
#     cmap="jet", 
#     s=20,        # Size of the points, adjust based on mesh density
#     edgecolors='none'
# )

# plt.colorbar(scatter, label="Absolute Difference in EXX Strain")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.title("Scatter Plot of Absolute Difference in EXX Strain at Element Centroids")
# plt.axis("equal") # Ensures the geometry isn't distorted
# plt.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.show()

# --- New plot with pcolormesh ---
plt.figure(figsize=(10, 8))

# Prepare data for pivoting
diff_df = pd.DataFrame({
    'X': a['X'],
    'Y': a['Y'],
    'plot': abs((s['EXY'] - a['EXY']))
})

# Pivot the dataframe to get a 2D grid of the difference values
diff_pivot = diff_df.pivot(index='Y', columns='X', values='plot')

x_coords = diff_pivot.columns.to_numpy()
y_coords = diff_pivot.index.to_numpy()
C = diff_pivot.to_numpy()

X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)

colormesh = plt.pcolormesh(X_mesh, Y_mesh, C, cmap="jet", shading='auto', rasterized=True)

plt.colorbar(colormesh, label="Absolute Difference in EXY Strain")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Colormesh of Absolute Difference in EXY Strain at Element Centroids")
plt.axis("equal") # Ensures the geometry isn't distorted
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()