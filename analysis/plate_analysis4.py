import numpy as np
import matplotlib.pyplot as plt
# from Plate import Plate
from Analysis import Analysis
from OptimizedPlate import OptimizedPlate

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

xi = 0.2      # X position of load [m]
eta = 0.3     # Y position of load [m]


space_res_x = 0.0005  # [m]
space_res_y = 0.0005  # [m]

# Initial N is not critical as Analysis will find the converged value
p = OptimizedPlate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, 10)
ana = Analysis(p, 1e-7)

# %% Plot Strain exx
# Note: MATLAB script title was for epsilon_xy, but it plotted exx. Corrected here.
X_mesh, Y_mesh = np.meshgrid(p.x, p.y, indexing='xy')
plt.figure(num='Strain exx', figsize=(10, 8))
# The analysis object stores strains as (Nx, Ny), pcolormesh needs (Ny, Nx)
plt.pcolormesh(X_mesh, Y_mesh, ana.exx.T, cmap='plasma', shading='auto', rasterized=True)
plt.axis('tight')
plt.axis('equal')
plt.title(f'Converged Normal Strain $\\varepsilon_{{xx}}$ (N = {ana.N})')
plt.xlabel('$x \\left[\\mathrm{m}\\right]$')
plt.ylabel('$y \\left[\\mathrm{m}\\right]$')
plt.colorbar(label='$\\varepsilon_{xx}$')

# %% Plot Spectral Analysis of Strain exy
plt.figure(num='Spectral Analysis Strain exy', figsize=(10, 8))
# The FFT fields are (Ny, Nx)
plt.pcolormesh(ana.kx, ana.ky, np.log10(1 + np.abs(ana.ft_exy)), cmap='plasma', shading='auto', rasterized=True)
plt.axis('tight')
plt.axis('equal')
plt.colorbar(label='$\\log_{10}(1 + |\\mathrm{FFT}|)$')
plt.xlabel('Wavenumber $k_x \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.ylabel('Wavenumber $k_y \\; \\left[\\mathrm{m}^{-1}\\right]$')
plt.title('Spectrum of Strain $\\varepsilon_{xy}$ (Spatial Domain)')

# %% Plot Spectral Section along ky (at kx=0)
# Note: Corrected from suspected bug in MATLAB script's plotting variables
# %% Spectral Sections for all strains with Cutoff markers
center_ky_idx = len(ana.ky) // 2

strains_data = [
    (ana.ft_exx, ana.find_sampling_freq_exx, r'\varepsilon_{xx}'),
    (ana.ft_eyy, ana.find_sampling_freq_eyy, r'\varepsilon_{yy}'),
    (ana.ft_exy, ana.find_sampling_freq_exy, r'\varepsilon_{xy}')
]

for ft_field, freq_func, tex_label in strains_data:
    # Calculate cutoff using Analysis class functions
    fs_x, fs_y = freq_func()
    
    plt.figure(figsize=(10, 6))
    # Slice along kx at ky=0 (Wavenumber zero in Y)
    spectrum_slice = np.log10(1 + np.abs(ft_field[center_ky_idx, :]))
    plt.plot(ana.kx, spectrum_slice, linewidth=2, label=f'Spectrum ${tex_label}$')
    
    # Mark 99% energy cutoff frequency
    plt.axvline(x=fs_x, color='r', linestyle='--', alpha=0.7, label=f'99% Energy Cutoff ($k_x={fs_x:.2f}$)')
    plt.axvline(x=-fs_x, color='r', linestyle='--', alpha=0.7)
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.xlabel(r'Wavenumber $k_x \ [\mathrm{m}^{-1}]$')
    plt.ylabel(r'Amplitude $\log_{10}(1+|FT|)$')
    plt.title(f'Spectral Section along $k_x$ (at $k_y=0$) for ${tex_label}$')
    plt.legend()
    plt.xlim([-fs_x * 3.5, fs_x * 3.5])
    plt.tight_layout()

plt.show()