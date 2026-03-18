import numpy as np
import matplotlib.pyplot as plt
from Plate import Plate
from Analysis import Analysis

# Set matplotlib to use LaTeX for text rendering to match MATLAB script
# Note: This requires a LaTeX distribution to be installed on your system.
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
except Exception as e:
    print(f"Could not enable LaTeX rendering: {e}\nFalling back to default.")

D = 1.0       # Flexural rigidity [Nm^2]
P = -1.0      # Concentrated load [N]
a = 1.0       # Length side X [m]
b = 1.0       # Length side Y [m]
t = 0.002     # Thickness side Z [m]

xi = 0.5      # X position of load [m]
eta = 0.5     # Y position of load [m]


space_res_x = 0.0005  # [m]
space_res_y = 0.0005  # [m]

# Initial N is not critical as Analysis will find the converged value
p = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, 10)
ana = Analysis(p, 1e-7)

# %% Plot Strain exx
# Note: MATLAB script title was for epsilon_xy, but it plotted exx. Corrected here.
X_mesh, Y_mesh = np.meshgrid(p.x, p.y, indexing='xy')
plt.figure(num='Strain exx', figsize=(10, 8))
# The analysis object stores strains as (Nx, Ny), pcolormesh needs (Ny, Nx)
plt.pcolormesh(X_mesh, Y_mesh, ana.exx.T, cmap='jet', shading='auto', rasterized=True)
plt.axis('tight')
plt.axis('equal')
plt.title(f'Converged Normal Strain $\\varepsilon_{{xx}}$ (N = {ana.N})')
plt.xlabel('$x \\left[\\mathrm{m}\\right]$')
plt.ylabel('$y \\left[\\mathrm{m}\\right]$')
plt.colorbar(label='$\\varepsilon_{xx}$')

# %% Plot Spectral Analysis of Strain exy
plt.figure(num='Spectral Analysis Strain exy', figsize=(10, 8))
# The FFT fields are (Ny, Nx)
plt.pcolormesh(ana.kx, ana.ky, np.log10(1 + np.abs(ana.ft_exy)), cmap='jet', shading='auto', rasterized=True)
plt.axis('tight')
plt.axis('equal')
plt.colorbar(label='$\\log_{10}(1 + |\\mathrm{FFT}|)$')
plt.xlabel('Wavenumber $k_x \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.ylabel('Wavenumber $k_y \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.title('Spectrum of Strain $\\varepsilon_{xy}$ (Spatial Domain)')

# %% Plot Spectral Section along ky (at kx=0)
# Note: Corrected from suspected bug in MATLAB script's plotting variables
center_kx_idx = len(ana.kx) // 2
plt.figure(num='Spectral Section ky', figsize=(10, 6))
plt.plot(ana.ky, np.log10(1 + np.abs(ana.ft_exy[:, center_kx_idx])), linewidth=2)
plt.grid(True)
plt.xlabel('Wavenumber $k_y \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.ylabel('Amplitude $\\log_{10}(1+|\\mathrm{FFT}|)$')
plt.title('Spectrum Slice along $k_y$ (at $k_x=0$)')

# %% Plot Spectral Section along kx (at ky=0)
# Note: Corrected from suspected bug in MATLAB script's plotting variables
center_ky_idx = len(ana.ky) // 2
plt.figure(num='Spectral Section kx', figsize=(10, 6))
plt.plot(ana.kx, np.log10(1 + np.abs(ana.ft_exy[center_ky_idx, :])), linewidth=2)
plt.grid(True)
plt.xlabel('Wavenumber $k_x \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.ylabel('Amplitude $\\log_{10}(1+|\\mathrm{FFT}|)$')
plt.title('Spectrum Slice along $k_x$ (at $k_y=0$)')
plt.xlim([ana.kx[0], ana.kx[-1]])

# %% Final Analysis Call
# This method prints the results to the console
ana.find_sampling_freq_exy()

plt.show()