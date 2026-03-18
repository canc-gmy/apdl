import numpy as np
import matplotlib.pyplot as plt
from Analysis import Analysis
import os

# Set matplotlib to use LaTeX for text rendering
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
except Exception as e:
    print(f"Could not enable LaTeX rendering: {e}\nFalling back to default.")

# --- Configuration ---
# Define the path to the CSV file you want to analyze.
# This file should be generated from Plate.py, for example using plate_analysis1.py
CSV_FILE_PATH = r"C:\Users\rossi\Documents\tesi\apdl\csv\ana_0.010_0.010.csv"

if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"The specified CSV file was not found: {CSV_FILE_PATH}\n"
                            "Please run plate_analysis1.py or another script to generate it.")

ana = Analysis(CSV_FILE_PATH)


# --- Plotting ---

def plot_spectrum(ft_data, kx, ky, title, cbar_label='$\\log_{10}(1 + |\\mathrm{FFT}|)$'):
    """Helper function to plot the 2D spectrum of a given field."""
    plt.figure(figsize=(10, 8))
    # The FFT fields are (Ny, Nx) which matches the pcolormesh requirement with kx (Nx,) and ky (Ny,)
    plt.pcolormesh(kx, ky, np.log10(1 + np.abs(ft_data)), cmap='plasma', shading='auto', rasterized=True)
    plt.axis('tight')
    plt.axis('equal')
    plt.colorbar(label=cbar_label)
    plt.xlabel('Wavenumber $k_x \\; \\left[\\mathrm{m}^{-1}\\right]$')
    plt.ylabel('Wavenumber $k_y \\; \\left[\\mathrm{m}^{-1}\\right]$')
    plt.title(title)

# Plot the spectrum for each quantity
plot_spectrum(ana.ft_w, ana.kx, ana.ky, 'Spectrum of Deflection $w$')
plot_spectrum(ana.ft_exx, ana.kx, ana.ky, 'Spectrum of Strain $\\varepsilon_{xx}$')
plot_spectrum(ana.ft_eyy, ana.kx, ana.ky, 'Spectrum of Strain $\\varepsilon_{yy}$')
plot_spectrum(ana.ft_exy, ana.kx, ana.ky, 'Spectrum of Strain $\\varepsilon_{xy}$')

# Show all the plots
plt.show()