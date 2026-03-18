import numpy as np
import matplotlib.pyplot as plt
from Plate import Plate
from Analysis import Analysis

# %% Physical and Geometrical Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0
a = 1.0
b = 1.0
t = 0.002

# Initial load position (will be varied in the loop)
xi = 0.5
eta = 0.5

space_res_x = 0.001
space_res_y = 0.001

# Define the grid of load positions to test
# The original script explores a quarter of the plate, from (0,0) to (0.5, 0.5)
# excluding the boundaries.
xi_positions = np.arange(0.05, 0.501, 0.05)
eta_positions = np.arange(0.05, 0.501, 0.05)
sample_x = len(xi_positions)
sample_y = len(eta_positions)
freq_x = np.zeros((sample_x, sample_y))
freq_y = np.zeros((sample_x, sample_y))


for i, xi_val in enumerate(xi_positions):
    for j, eta_val in enumerate(eta_positions):
        ana = Analysis(f"grid/grid_{xi_val:.3f}_{eta_val:.3f}.parquet")
        print(f'Running analysis for load at: ({xi_val:.3f}, {eta_val:.3f})')
        # find_sampling_freq_exx returns critical wavenumbers for x and y directions
        fx, fy = ana.find_sampling_freq_exx()
        freq_x[i, j] = fx
        freq_y[i, j] = fy

# %% Plotting results

# Create meshgrid for plotting
Eta_mesh, Xi_mesh = np.meshgrid(eta_positions, xi_positions)

# Calculate recommended mesh size in mm from the critical wavenumber
mesh_size_x_mm = 1000 / (2 * freq_x)

plt.figure(num='Sampling Frequency Distribution X', figsize=(12, 10))
plt.pcolormesh(Eta_mesh, Xi_mesh, mesh_size_x_mm, cmap='plasma', shading='auto')
plt.colorbar(label='Max mesh size [mm]')

# Add text annotations for the mesh size values
for i in range(sample_x):
    for j in range(sample_y):
        plt.text(eta_positions[j], xi_positions[i], f'{mesh_size_x_mm[i, j]:.1f}',
                 ha='center', va='center', color='black', fontsize=7)

plt.xlabel('Load Y position [m]')
plt.ylabel('Load X position [m]')
plt.title('Recommended Maximum Mesh Size [mm] for $\\epsilon_{xx}$ (based on $k_x$)')
plt.axis('equal')
plt.tight_layout()
plt.show()


# Calculate recommended mesh size in mm from the critical wavenumber
mesh_size_y_mm = 1000 / (2 * freq_y)

plt.figure(num='Sampling Frequency Distribution Y', figsize=(12, 10))
plt.pcolormesh(Eta_mesh, Xi_mesh, mesh_size_y_mm, cmap='plasma', shading='auto')
plt.colorbar(label='Max mesh size [mm]')

# Add text annotations for the mesh size values
for i in range(sample_x):
    for j in range(sample_y):
        plt.text(eta_positions[j], xi_positions[i], f'{mesh_size_y_mm[i, j]:.1f}',
                 ha='center', va='center', color='black', fontsize=7)

plt.xlabel('Load Y position [m]')
plt.ylabel('Load X position [m]')
plt.title('Recommended Maximum Mesh Size [mm] for $\\epsilon_{xx}$ (based on $k_y$)')
plt.axis('equal')
plt.tight_layout()
plt.show() 