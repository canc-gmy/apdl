import numpy as np
import matplotlib.pyplot as plt
from Plate import Plate
# from analysis import Analysis # Ensure Analysis exists if you intend to run it

# Parameters
D = 1.0                # Flexural rigidity [Nm^2]
P = -1.0               # Concentrated load [N]
a = 1.0                # X length of plate [m]
b = 1.0                # Y length of plate [m]
t = 0.001              # Thickness [m]

xi = 0.5               # X position of load [m]
eta = 0.5              # Y position of load [m]

space_res_x = 0.005    # Spatial resolution X direction of results [m]
space_res_y = 0.005    # Spatial resolution Y direction of results [m]

# %% Find a suitable N for series precision
res = []

for N in range(5, 35, 5): # 5, 10, 15, 20, 25, 30
    plate = Plate(a, b, t, space_res_x, space_res_y, xi, eta, D, P, N)
    res.append(plate.deflection())

# %% Specific Calculation for N = 15
N = 15
p = Plate(a, b, t, space_res_x, space_res_y, 0.1, 0.5, D, P, N)

w = p.deflection()
exx = p.strain_xx()
eyy = p.strain_yy()
exy = p.strain_xy()

# %% Spectral Transforms
ft_w = np.fft.fftshift(np.fft.fft2(w.T))
ft_exx = np.fft.fftshift(np.fft.fft2(exx.T))
ft_eyy = np.fft.fftshift(np.fft.fft2(eyy.T))
ft_exy = np.fft.fftshift(np.fft.fft2(exy.T))

spatial_freq_x = 1 / p.space_res_x
spatial_freq_y = 1 / p.space_res_y

# Generate coordinate mappings matching the MATLAB script
X_freq = spatial_freq_x / p.a * (p.x - p.a / 2)
Y_freq = spatial_freq_y / p.b * (p.y - p.b / 2)
X_mesh, Y_mesh = np.meshgrid(X_freq, Y_freq)

# %% 3D Surface Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_mesh, Y_mesh, np.real(ft_exx), edgecolor='white')
ax.set_xlabel('X freq')
ax.set_ylabel('Y freq')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
plt.show()