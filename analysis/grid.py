import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 0.5, 0.01)
y = np.arange(0, 0.5, 0.01)
xi = 0.1
eta = 0.1
a = 1
b = 1
eps = np.zeros((len(y), len(x)))
M_range = np.arange(1, 1000)
N_range = np.arange(1, 1000)

# Chunked vectorized implementation.
# This approach processes data in blocks to be memory efficient while
# maintaining vectorization speed for inner loops.

# Shapes: (M,) and (N,)
m_pi_a = M_range * np.pi / a
n_pi_b = N_range * np.pi / b

# Shapes: (M, X) and (N, Y)
sin_mx_sq = np.sin(np.outer(m_pi_a, x))**2
cos_mx_sq = np.cos(np.outer(m_pi_a, x))**2
sin_ny_sq = np.sin(np.outer(n_pi_b, y))**2
cos_ny_sq = np.cos(np.outer(n_pi_b, y))**2

# --- Calculate Coefficients ---
# Shape: (M, N)
m2_a2 = (M_range/a)**2
n2_b2 = (N_range/b)**2
denominator = (m2_a2[:, None] + n2_b2[None, :])**2
sin_m_xi = np.sin(M_range * np.pi * xi / a)
sin_n_eta = np.sin(N_range * np.pi * eta / b)
coeffs = (sin_m_xi[:, None] * sin_n_eta[None, :]) / denominator

# --- Calculate the term inside the square root and Summation ---
# Using a chunked approach to avoid Memory Error with large 4D tensors.
m4_a4 = m2_a2**2
n4_b4 = n2_b2**2

chunk_size = 100
for i in range(0, len(M_range), chunk_size):
    m_slice = slice(i, i + chunk_size)
    # Get subset of M-dependent arrays
    m2_sub = m2_a2[m_slice]
    m4_sub = m4_a4[m_slice]
    sin_mx_sub = sin_mx_sq[m_slice]
    cos_mx_sub = cos_mx_sq[m_slice]

    for j in range(0, len(N_range), chunk_size):
        n_slice = slice(j, j + chunk_size)
        # Get subset of N-dependent arrays
        n2_sub = n2_b2[n_slice]
        n4_sub = n4_b4[n_slice]
        sin_ny_sub = sin_ny_sq[n_slice]
        cos_ny_sub = cos_ny_sq[n_slice]

        coeffs_sub = coeffs[m_slice, n_slice]

        # Calculate terms for this block (M_sub, N_sub, Y, X)
        term_A_chunk = (m4_sub[:, None] + n4_sub[None, :])[:, :, None, None] * \
                       sin_ny_sub[None, :, :, None] * \
                       sin_mx_sub[:, None, None, :]

        term_B_chunk = (m2_sub[:, None] * n2_sub[None, :])[:, :, None, None] * \
                       cos_ny_sub[None, :, :, None] * \
                       cos_mx_sub[:, None, None, :]

        inside_sqrt_chunk = np.pi**4 * (term_A_chunk + term_B_chunk)

        # Accumulate result
        eps += np.sum(coeffs_sub[:, :, None, None] * np.sqrt(inside_sqrt_chunk), axis=(0, 1))

def plot_eps(x_vals, y_vals, eps_matrix):
    X, Y = np.meshgrid(x_vals, y_vals)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, eps_matrix, cmap='viridis', shading='auto')
    plt.colorbar(label='$\\epsilon$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Epsilon Colormap')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

plot_eps(x, y, eps)
