import numpy as np
import matplotlib.pyplot as plt
from Plate import Plate

# This script is a Python conversion of the MATLAB script plate_analysis2.
# It performs a convergence study for a plate under a concentrated load,
# and then conducts a spectral analysis on the resulting strain fields.

# %% Physical and Geometrical Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0               # Concentrated load [N]
a = 1.0                # X length of plate [m]
b = 1.0                # Y length of plate [m]
t = 0.01               # Thickness [m]

xi = 0.6               # X position of load [m]
eta = 0.6              # Y position of load [m]

# Spatial resolution
space_res_x = 0.001
space_res_y = 0.001

# %% Convergence Analysis
TOL = 2e-7  # Convergence threshold based on sensor sensitivity (0.1 microstrain)
N = 300
max_diff = np.inf

print('Starting convergence analysis for epsilon_xy...')

# Create a plate object to define dimensions and then initialize res_old
p = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, N)
res_old = np.zeros((len(p.x), len(p.y)))

# %% Automatic Convergence Loop
while max_diff > TOL:
    p.N = int(N)
    res_new = p.strain_xy()  # Most critical component

    # Calculate difference from previous step
    diff_matrix = np.abs(res_new - res_old)

    # Exclude the load point (singularity) for Max Error calculation
    X, Y = np.meshgrid(p.x, p.y, indexing='ij')
    dist_from_load = np.sqrt((X - xi)**2 + (Y - eta)**2)
    mask = dist_from_load > 0.05

    max_diff = np.max(diff_matrix[mask])

    print(f'N = {N} | Max Delta Epsi_xy (away from load): {max_diff:.2e}')

    res_old = res_new
    if max_diff > TOL:
        N += 50  # Increment step

print(f'Convergence reached at N = {N} with error < {TOL:.2e}')
N_final = N

# %% Plot of Converged XY Strain
plt.figure(figsize=(8, 6))
X_mesh, Y_mesh = np.meshgrid(p.x, p.y, indexing='xy')
plt.pcolormesh(X_mesh, Y_mesh, res_old.T, cmap='jet', shading='auto')
plt.title(f'Converged Shear Strain $\\epsilon_{{xy}}$ (N = {N_final})')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='$\\epsilon_{xy}$')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% Generate Strains for Spectral Analysis
print(f"Recalculating fields with final N = {N_final}")
p = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, N_final)

# Calculate fields
exx = p.strain_xx()
eyy = p.strain_yy()

plt.figure(figsize=(8, 6))
plt.pcolormesh(X_mesh, Y_mesh, exx.T, cmap='jet', shading='auto')
plt.title(f'Converged Strain $\\epsilon_{{xx}}$ (N = {N_final})')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='$\\epsilon_{xx}$')
plt.axis('equal')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.pcolormesh(X_mesh, Y_mesh, eyy.T, cmap='jet', shading='auto')
plt.title(f'Converged Strain $\\epsilon_{{yy}}$ (N = {N_final})')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='$\\epsilon_{yy}$')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% 2D Fourier Transform (FFT)
# The python Plate class returns strain fields as (Nx, Ny).
# We transpose to (Ny, Nx) for fft2, which is consistent with how
# meshgrid and pcolormesh work with (x,y) coordinates.
ft_exx = np.fft.fftshift(np.fft.fft2(exx.T))

# Spatial frequency axes (Wavenumbers k)
Fs_x = 1 / space_res_x
Fs_y = 1 / space_res_y
kx = np.linspace(-Fs_x/2, Fs_x/2, len(p.x))
ky = np.linspace(-Fs_y/2, Fs_y/2, len(p.y))

# Amplitude Spectrum
magnitude_exx = np.abs(ft_exx)

plt.figure(num='Spectral Analysis of Strain', figsize=(8, 6))
# magnitude_exx is (Ny, Nx), kx is (Nx,), ky is (Ny,)
plt.pcolormesh(kx, ky, np.log10(magnitude_exx), cmap='jet', shading='auto')
cbar = plt.colorbar()
cbar.set_label('log10(|FFT|)')
plt.xlabel('Wavenumber kx [1/m]')
plt.ylabel('Wavenumber ky [1/m]')
plt.title('Strain Spectrum (Spatial Frequency Domain)')
plt.show()

# %% Central Section of the Spectrum (1D Plot)
# Integer indices for center of spectrum
cx = len(kx) // 2
cy = len(ky) // 2

plot_range = 40
idx_range = slice(cx - plot_range, cx + plot_range + 1)

plt.figure(num='Spectral Section', figsize=(10, 6))
# magnitude_exx is (Ny, Nx), so we index with [cy, idx_range] to get a slice along kx
plt.plot(kx[idx_range], np.log10(magnitude_exx[cy, idx_range]), linewidth=2)
plt.grid(True)
plt.xlabel('Wavenumber kx [1/m]')
plt.ylabel('log10(|FFT| Amplitude)')
plt.title('Spectrum slice along kx (at ky=0)')
plt.show()

# %% Cutoff Frequency Test (99% Energy)
psd_exx = np.abs(ft_exx)**2

# Energy profile along kx: sum over ky for each kx.
# psd_exx is (Ny, Nx), so sum along axis 0.
energy_profile_x = np.sum(psd_exx, axis=0)
cumulative_energy_x = np.cumsum(energy_profile_x) / np.sum(energy_profile_x)

threshold = 0.99
# Find first index where cumulative energy exceeds threshold
idx_cutoff = np.argmax(cumulative_energy_x >= threshold)
k_star = np.abs(kx[idx_cutoff])

# Max mesh size (Nyquist criterion)
h_max = 1 / (2 * k_star)

print('--- SPECTRAL ANALYSIS RESULTS ---')
print(f'Critical wavenumber (99% energy): {k_star:.2f} 1/m')
print(f'Recommended maximum mesh size: {h_max * 1000:.2f} mm')


