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

data = pd.read_parquet(r"C:\Users\rossi\Documents\tesi\apdl\sim\sim_0.006_0.024.parquet")
t = 0.001

delta = 0.002
Y = 0.035
line_data = data.query(f'abs(((X - 0.001)) % @delta < 1e-6 | abs((X - 0.001) % @delta - @delta) < 1e-6) & Y == @Y')
curvature_line = -line_data['EXX']/t
z_disp = np.cumsum(np.cumsum(curvature_line.values) * delta) * delta # double integral
z_disp = z_disp - z_disp[-1] / np.max(line_data['X'].unique()) * line_data['X'].unique() # boudary condition z[a] == 0


fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(line_data['X'].unique(), line_data['Z'], 
         label='Reference Deflection', 
         color='black', 
         linestyle='--', 
         linewidth=2)

# Plot 2: Calculated Data from Strain
ax.plot(line_data['X'].unique(), z_disp, 
         label='Calculated Deflection', 
         color='purple', 
         linewidth=2)

# Titles and Axis Labels
# (Note: Update 'm' to 'mm' or 'in' if your data uses different units)
ax.set_title('Plate line Deflection Profile: FEM vs. Strain-Derived' + f'($\\delta = {delta:.3f}; y = {Y:.3f})$', fontsize=14, pad=15)
ax.set_xlabel(r"$x$ Coordinate $\left[\mathrm{m}\right]$")
ax.set_ylabel(r"$z$ Coordinate $\left[\mathrm{m}\right]$")

# Grid and Legend
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Clean up layout margins and render
plt.tight_layout()
plt.show()