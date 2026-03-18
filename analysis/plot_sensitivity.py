import numpy as np
import matplotlib.pyplot as plt
from OptimizedPlate import OptimizedPlate
from pathlib import Path
import os

# This script creates a plot showing the sensitivity of a calculated strain metric
# to the position of a concentrated load on a plate.

# --- Parameters ---

# Plate parameters from other analysis scripts
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0               # Concentrated load [N]
a = 1.0                # X length of plate [m]
b = 1.0                # Y length of plate [m]
t = 0.002              # Thickness [m]
space_res_x = 0.005    # Spatial resolution X for results [m]
space_res_y = 0.005    # Spatial resolution Y for results [m]
N_series = 500         # Number of series terms for calculation, 500 is a balance
                       # between accuracy and computation time.

# Analysis parameters
delta_thr = 1e-6       # A small strain value for the sensitivity metric, e.g., 1 microstrain

# Grid for load positions (quarter plate is analyzed due to symmetry)
# We avoid the very edges (0.0) where results might be singular.
xi_points = np.linspace(0.05, 0.5, 10)
eta_points = np.linspace(0.05, 0.5, 10)

# Data storage
results = []

# --- Calculation Loop ---
print("Starting sensitivity analysis for different load positions...")

# Iterate over the grid of load positions
for eta_val in eta_points:
    for xi_val in xi_points:
        print(f"  Analyzing load at: (xi={xi_val:.3f}, eta={eta_val:.3f})")

        # Create a plate object for the given load position.
        # We use OptimizedPlate as it is faster for many calculations.
        plate = OptimizedPlate(a, b, t, space_res_x, space_res_y, xi_val, eta_val, D, P, N_series)

        # Get the dataframe with results at element centroids.
        # This is done for a quarter of the plate for efficiency.
        df = plate.dataframe_centroid

        mean_val = np.nan
        if not df.empty and ('EPS_ETA' in df.columns) and ('EPS_XI' in df.columns):
            # Calculate the sensitivity metric as requested by the user.
            # The metric is the mean of: threshold / |d(eps)/d(eta) + d(eps)/d(xi)|
            denominator = np.abs(df['EPS_ETA'] + df['EPS_XI'])
            
            # Avoid division by zero
            # We can replace zeros with nan
            denominator[denominator == 0] = np.nan

            sensitivity_metric = delta_thr / denominator

            # Calculate the mean, ignoring any NaNs that resulted from division by zero
            mean_val = np.nanmean(sensitivity_metric)

        # Store results
        results.append({'xi': xi_val, 'eta': eta_val, 'mean_val': mean_val})

print("Analysis complete.")

# --- Plotting ---
print("Generating plot...")

# Extract data for plotting
xi_coords = [r['xi'] for r in results]
eta_coords = [r['eta'] for r in results]
mean_values = np.array([r['mean_val'] for r in results])

# Filter out any NaN values that may have occurred
valid_indices = ~np.isnan(mean_values)
xi_coords = np.array(xi_coords)[valid_indices]
eta_coords = np.array(eta_coords)[valid_indices]
mean_values = mean_values[valid_indices]

# Scale the mean values for plotting circle sizes.
# The 's' parameter in scatter is in points^2. A scaling factor is needed for visibility.
if mean_values.size > 0 and np.nanmax(mean_values) > 0:
    # Scale sizes proportionally, with a base size to ensure small values are visible.
    sizes = 50 + (mean_values / np.nanmax(mean_values)) * 1500
    
    # Create a function to unscale sizes for the legend
    def unscale_func(s):
        return ((s - 50) / 1500) * np.nanmax(mean_values)
else:
    sizes = np.array([])

# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

if sizes.size > 0:
    scatter = ax.scatter(
        xi_coords, 
        eta_coords, 
        s=sizes, 
        c=mean_values, 
        cmap='viridis', 
        alpha=0.7, 
        edgecolors='black', 
        linewidth=0.5
    )

    # Create a colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Sensitivity Metric')

    # Create a legend for the circle sizes
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5, func=unscale_func)
    ax.legend(handles, [f'{float(l):.2e}' for l in labels], loc="center left", bbox_to_anchor=(1.05, 0.5), title="Metric Value")

ax.set_title('Sensitivity Metric vs. Load Position (xi, eta)')
ax.set_xlabel('Load X position (xi) [m]')
ax.set_ylabel('Load Y position (eta) [m]')
ax.set_aspect('equal', 'box')
# Set plot limits to show the quarter plate being analyzed
ax.set_xlim(0, a / 2 + 0.05)
ax.set_ylim(0, b / 2 + 0.05)

plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()