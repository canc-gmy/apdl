"""
This module provides an optimized implementation of the Plate class for
large-scale calculations.

It uses the Numba library to JIT-compile and parallelize the most
computationally intensive parts of the analysis, resulting in a significant
performance increase.

Requires: numpy, pandas, numba
"""
import numpy as np
import pandas as pd
import numba


@numba.jit(nopython=True, parallel=True, cache=True, boundscheck=False)
def _calculate_eps_variants_core(y_coords, x_coords, a, b, xi, eta, N,
                                 m_range, n_range,
                                 m2_a2, n2_b2, m4_a4, n4_b4, coeffs_eps):
    """
    Numba-accelerated core function to calculate eps variants.
    This function is JIT-compiled and parallelized for maximum performance.
    It's optimized by pre-calculating trigonometric values and other factors
    to reduce redundant computations inside the main loop.
    """
    eps_yx = np.zeros((len(y_coords), len(x_coords)))
    eps_xi_yx = np.zeros((len(y_coords), len(x_coords)))
    eps_eta_yx = np.zeros((len(y_coords), len(x_coords))) # Initialize, but will be overwritten
    
    pi = np.pi
    pi4 = pi**4

    # Precompute sin/cos terms for all x and m values.
    sin_mx_sq_grid = np.empty((N, len(x_coords)))
    cos_mx_sq_grid = np.empty((N, len(x_coords)))
    for m_idx in numba.prange(N):
        m = m_range[m_idx]
        m_pi_a = m * pi / a
        for i in range(len(x_coords)):
            m_pi_a_x = m_pi_a * x_coords[i]
            sin_mx_sq_grid[m_idx, i] = np.sin(m_pi_a_x)**2
            cos_mx_sq_grid[m_idx, i] = np.cos(m_pi_a_x)**2

    # Precompute sin/cos terms for all y and n values.
    sin_ny_sq_grid = np.empty((N, len(y_coords)))
    cos_ny_sq_grid = np.empty((N, len(y_coords)))
    for n_idx in numba.prange(N):
        n = n_range[n_idx]
        n_pi_b = n * pi / b
        for j in range(len(y_coords)):
            n_pi_b_y = n_pi_b * y_coords[j]
            sin_ny_sq_grid[n_idx, j] = np.sin(n_pi_b_y)**2
            cos_ny_sq_grid[n_idx, j] = np.cos(n_pi_b_y)**2

    # Precompute factors that only depend on m and n.
    term_A_factor = np.empty((N, N))
    term_B_factor = np.empty((N, N))
    for m_idx in numba.prange(N):
        for n_idx in range(N):
            term_A_factor[m_idx, n_idx] = m4_a4[m_idx] + n4_b4[n_idx]
            term_B_factor[m_idx, n_idx] = m2_a2[m_idx] * n2_b2[n_idx]

    # Numba parallelizes this outer loop across all available CPU cores.
    for j in numba.prange(len(y_coords)):
        for i in range(len(x_coords)):
            total_eps = 0.0
            # total_eps_xi = 0.0 # No longer needed
            # total_eps_eta = 0.0 # No longer needed

            # Inner loops over the series use pre-calculated values.
            for m_idx in range(N):
                sin_mx_sq = sin_mx_sq_grid[m_idx, i]
                cos_mx_sq = cos_mx_sq_grid[m_idx, i]

                for n_idx in range(N):
                    sin_ny_sq = sin_ny_sq_grid[n_idx, j]
                    cos_ny_sq = cos_ny_sq_grid[n_idx, j]

                    term_A = term_A_factor[m_idx, n_idx] * sin_ny_sq * sin_mx_sq
                    term_B = term_B_factor[m_idx, n_idx] * cos_ny_sq * cos_mx_sq

                    sqrt_term = np.sqrt(pi4 * (term_A + term_B))
                    total_eps += coeffs_eps[m_idx, n_idx] * sqrt_term

            eps_yx[j, i] = total_eps

    return eps_yx, np.zeros_like(eps_yx), np.zeros_like(eps_yx) # Return zeros for the unused ones


class OptimizedPlate:
    """
    An optimized version of the Timoshenko/Kirchhoff plate analyzer.

    This class uses Numba to accelerate the most demanding calculations,
    making it suitable for high-resolution grids and a large number of
    series terms (N).
    """

    def __init__(self, a=0.5, b=0.5, t=0.001, space_res_x=0.01, space_res_y=0.01,
                 xi=0.5, eta=0.5, D=1.0, P=-1.0, N=100):
        self._a = a
        self._b = b
        self._t = t
        self._space_res_x = space_res_x
        self._space_res_y = space_res_y
        self._xi = xi
        self._eta = eta
        self._D = D
        self._P = P
        self._N = int(N)
        self._dataframe = None
        self._dataframe_centroid = None

        self._recompute_grid()

    def _recompute_grid(self):
        """Recalculates the spatial grid for the plate."""
        self.x = np.linspace(0, self._a, int(round(self._a / self._space_res_x)) + 1)
        self.y = np.linspace(0, self._b, int(round(self._b / self._space_res_y)) + 1)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidates the cached dataframe."""
        self._dataframe = None
        self._dataframe_centroid = None

    @property
    def dataframe(self):
        """
        Provides the plate analysis results as a pandas DataFrame.
        The DataFrame is cached and recomputed only when plate parameters change.
        """
        if self._dataframe is None:
            w, exx, eyy, exy = self.calculate_plate_state()

            exx_eta = self.strain_xx_eta()
            exx_xi = self.strain_xx_xi()
            eps, eps_xi, eps_eta = self.calculate_eps_variants()

            # meshgrid with 'xy' indexing (default) creates grids that are
            # (Ny, Nx). The calculated fields are (Nx, Ny), so they must
            # be transposed before flattening to match.
            X, Y = np.meshgrid(self.x, self.y)

            df = pd.DataFrame({
                'X': X.flatten(),
                'Y': Y.flatten(),
                'Z': w.T.flatten(),
                'EXX': exx.T.flatten(),
                'EYY': eyy.T.flatten(),
                'EXY': exy.T.flatten(),
                'EXX_ETA': exx_eta.T.flatten(),
                'EXX_XI': exx_xi.T.flatten(),
                'EPS': eps.T.flatten(),
                'EPS_XI': eps_xi.T.flatten(),
                'EPS_ETA': eps_eta.T.flatten(),
            })
            df.index = pd.Series(np.arange(1, len(df) + 1), name="Element ID")
            self._dataframe = df
        return self._dataframe

    @property
    def dataframe_centroid(self):
        """
        Provides plate analysis results as a pandas DataFrame with values
        calculated directly at the centroid of each grid cell.
        """
        if self._dataframe_centroid is None:
            original_x, original_y = self.x, self.y
            try:
                centroid_x = (original_x[:-1] + original_x[1:]) / 2.0
                centroid_y = (original_y[:-1] + original_y[1:]) / 2.0

                self.x = centroid_x
                self.y = centroid_y

                w_c, exx_c, eyy_c, exy_c = self.calculate_plate_state()

                X_c, Y_c = np.meshgrid(centroid_x, centroid_y, indexing='ij')
                num_elements = len(centroid_x) * len(centroid_y)
                element_ids = np.arange(1, num_elements + 1)

                df = pd.DataFrame({
                    'X': X_c.flatten(order="F"),
                    'Y': Y_c.flatten(order="F"),
                    'Z': w_c.flatten(order="F"),
                    'EXX': exx_c.flatten(order="F"),
                    'EYY': eyy_c.flatten(order="F"),
                    'EXY': exy_c.flatten(order="F"),
                })
                df.index = pd.Series(element_ids, name="Element ID")
                self._dataframe_centroid = df
            finally:
                self.x, self.y = original_x, original_y
        return self._dataframe_centroid

    @property
    def dataframe_centroid_with_eps(self):
        """
        Provides plate analysis results as a pandas DataFrame with values
        calculated directly at the centroid of each grid cell.
        """
        if self._dataframe_centroid is None:
            original_x, original_y = self.x, self.y
            try:
                centroid_x = (original_x[:-1] + original_x[1:]) / 2.0
                centroid_y = (original_y[:-1] + original_y[1:]) / 2.0

                self.x = centroid_x
                self.y = centroid_y

                w_c, exx_c, eyy_c, exy_c, eps_c, eps_xi_c, eps_eta_c = self.calculate_plate_state_with_eps()

                X_c, Y_c = np.meshgrid(centroid_x, centroid_y, indexing='ij')
                num_elements = len(centroid_x) * len(centroid_y)
                element_ids = np.arange(1, num_elements + 1)

                df = pd.DataFrame({
                    'X': X_c.flatten(order="F"),
                    'Y': Y_c.flatten(order="F"),
                    'Z': w_c.flatten(order="F"),
                    'EXX': exx_c.flatten(order="F"),
                    'EYY': eyy_c.flatten(order="F"),
                    'EXY': exy_c.flatten(order="F"),
                    'EPS': eps_c.flatten(order="F"),
                    'EPS_XI': eps_xi_c.flatten(order="F"),
                    'EPS_ETA': eps_eta_c.flatten(order="F"),
                })
                df.index = pd.Series(element_ids, name="Element ID")
                self._dataframe_centroid = df
            finally:
                self.x, self.y = original_x, original_y
        return self._dataframe_centroid

    @property
    def a(self): return self._a
    @a.setter
    def a(self, value): self._a = value; self._recompute_grid()

    @property
    def b(self): return self._b
    @b.setter
    def b(self, value): self._b = value; self._recompute_grid()

    @property
    def t(self): return self._t
    @t.setter
    def t(self, value): self._t = value; self._invalidate_cache()

    @property
    def space_res_x(self): return self._space_res_x
    @space_res_x.setter
    def space_res_x(self, value): self._space_res_x = value; self._recompute_grid()

    @property
    def space_res_y(self): return self._space_res_y
    @space_res_y.setter
    def space_res_y(self, value): self._space_res_y = value; self._recompute_grid()

    @property
    def xi(self): return self._xi
    @xi.setter
    def xi(self, value): self._xi = value; self._invalidate_cache()

    @property
    def eta(self): return self._eta
    @eta.setter
    def eta(self, value): self._eta = value; self._invalidate_cache()

    @property
    def D(self): return self._D
    @D.setter
    def D(self, value): self._D = value; self._invalidate_cache()

    @property
    def P(self): return self._P
    @P.setter
    def P(self, value): self._P = value; self._invalidate_cache()

    @property
    def N(self): return self._N
    @N.setter
    def N(self, value): self._N = int(value); self._invalidate_cache()

    def deflection(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        scalar_factor = (4 * self.P) / (np.pi**4 * self.a * self.b * self.D)
        numerator_load = np.sin(m * np.pi * self.xi / self.a) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator
        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))
        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_xx(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        scalar_factor = (self.t / 2) * (4 * self.P) / (np.pi**2 * self.a**3 * self.b * self.D)
        numerator_load = (m**2 * np.sin(m * np.pi * self.xi / self.a)) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator
        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))
        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_yy(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        scalar_factor = (self.t / 2) * (4 * self.P) / (np.pi**2 * self.a * self.b**3 * self.D)
        numerator_load = (n**2 * np.sin(m * np.pi * self.xi / self.a)) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator
        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))
        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_xy(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        scalar_factor = -(self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a**2 * self.b**2 * self.D)
        numerator_load = np.sin(m * np.pi * self.xi / self.a) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (numerator_load / denominator) * (m * n)
        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))
        return scalar_factor * (cos_x @ C @ cos_y.T)

    def compute_strain_xy_segment(self, m_range, n_range):
        if len(m_range) == 0 or len(n_range) == 0: # This line has a bug, it should be self.eta / self.b
            return 0 # This line has a bug, it should be self.eta / self.b
        m = np.array(m_range)[:, np.newaxis]
        n = np.array(n_range)[np.newaxis, :]
        load_term = np.sin(m * np.pi * self.xi / self.a) * np.sin(n * np.pi * self.eta / self.b)
        den = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (load_term / den) * (m * n)
        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))
        return cos_x @ C @ cos_y.T

    def strain_xy_delta(self, N_start, N_end):
        if N_start >= N_end:
            return np.zeros((len(self.x), len(self.y)))
        scalar_factor = -self.t * (4 * self.P) / (np.pi**2 * self.a**2 * self.b**2 * self.D)
        m_new = np.arange(N_start + 1, N_end + 1)
        n_full = np.arange(1, N_end + 1)
        delta_1 = self.compute_strain_xy_segment(m_new, n_full)
        m_old = np.arange(1, N_start + 1)
        n_new = np.arange(N_start + 1, N_end + 1)
        delta_2 = self.compute_strain_xy_segment(m_old, n_new)
        return scalar_factor * (delta_1 + delta_2)

    def calculate_plate_state(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        num_x, num_y = len(self.x), len(self.y)
        base_factor = (4 * self.P) / (np.pi**4 * self.a * self.b * self.D)
        load_term = np.sin(m * np.pi * self.xi / self.a) * np.sin(n * np.pi * self.eta / self.b)
        den = ((m / self.a)**2 + (n / self.b)**2)**2
        C_base = load_term / den
        C_w = C_base
        C_xx = C_base * (m**2) * (np.pi**2 / self.a**2) * (self.t / 2)
        C_yy = C_base * (n**2) * (np.pi**2 / self.b**2) * (self.t / 2)
        C_xy = C_base * (m * n) * (np.pi**2 / (self.a * self.b)) * (-self.t / 2)
        S_y_sin = np.sin(n.flatten() * np.pi / self.b * self.y[:, np.newaxis]).T
        S_y_cos = np.cos(n.flatten() * np.pi / self.b * self.y[:, np.newaxis]).T
        F_w = C_w @ S_y_sin
        F_xx = C_xx @ S_y_sin
        F_yy = C_yy @ S_y_sin
        F_xy = C_xy @ S_y_cos
        w = np.zeros((num_x, num_y))
        e_xx = np.zeros((num_x, num_y))
        e_yy = np.zeros((num_x, num_y))
        e_xy = np.zeros((num_x, num_y))
        blockSize = 1000
        for i in range(0, num_x, blockSize):
            idx = slice(i, min(i + blockSize, num_x))
            x_slice = self.x[idx]
            S_x_sin_block = np.sin(x_slice[:, np.newaxis] * (m.flatten() * np.pi / self.a))
            S_x_cos_block = np.cos(x_slice[:, np.newaxis] * (m.flatten() * np.pi / self.a))
            w[idx, :] = base_factor * (S_x_sin_block @ F_w)
            e_xx[idx, :] = base_factor * (S_x_sin_block @ F_xx)
            e_yy[idx, :] = base_factor * (S_x_sin_block @ F_yy)
            e_xy[idx, :] = base_factor * (S_x_cos_block @ F_xy)
        return w, e_xx, e_yy, e_xy

    def calculate_plate_state_with_eps(self):
        """
        Calcola w, strain, modulo dello strain e derivate parziali 
        sull'intera griglia spaziale in modo vettorizzato a blocchi.
        Assume che self.x, self.y, self.xi, self.eta e self.t 
        siano stati definiti nell'istanza della classe.
        """
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]
        num_x, num_y = len(self.x), len(self.y)

        pi = np.pi
        base_factor = (4 * self.P) / (pi**4 * self.a * self.b * self.D)

        m_pi_a = m * pi / self.a
        n_pi_b = n * pi / self.b

        s_m_xi, c_m_xi = np.sin(m_pi_a * self.xi), np.cos(m_pi_a * self.xi)
        s_n_eta, c_n_eta = np.sin(n_pi_b * self.eta), np.cos(n_pi_b * self.eta)

        den = ((m / self.a)**2 + (n / self.b)**2)**2

        C_base = (s_m_xi * s_n_eta) / den
        C_xi = (m_pi_a * c_m_xi * s_n_eta) / den
        C_eta = (n_pi_b * s_m_xi * c_n_eta) / den

        k_xx = (m**2) * (pi**2 / self.a**2) * (self.t / 2)
        k_yy = (n**2) * (pi**2 / self.b**2) * (self.t / 2)
        k_xy = (m * n) * (pi**2 / (self.a * self.b)) * (-self.t / 2)

        S_y_sin = np.sin(n_pi_b.flatten() * self.y[:, np.newaxis]).T
        S_y_cos = np.cos(n_pi_b.flatten() * self.y[:, np.newaxis]).T

        F_w = C_base @ S_y_sin
        
        F_xx, F_yy, F_xy = (C_base * k_xx) @ S_y_sin, (C_base * k_yy) @ S_y_sin, (C_base * k_xy) @ S_y_cos
        F_exx_xi, F_eyy_xi, F_exy_xi = (C_xi * k_xx) @ S_y_sin, (C_xi * k_yy) @ S_y_sin, (C_xi * k_xy) @ S_y_cos
        F_exx_eta, F_eyy_eta, F_exy_eta = (C_eta * k_xx) @ S_y_sin, (C_eta * k_yy) @ S_y_sin, (C_eta * k_xy) @ S_y_cos

        w = np.zeros((num_x, num_y))
        e_xx = np.zeros((num_x, num_y))
        e_yy = np.zeros((num_x, num_y))
        e_xy = np.zeros((num_x, num_y))
        exx_xi, eyy_xi, exy_xi = np.zeros_like(e_xx), np.zeros_like(e_xx), np.zeros_like(e_xx)
        exx_eta, eyy_eta, exy_eta = np.zeros_like(e_xx), np.zeros_like(e_xx), np.zeros_like(e_xx)

        blockSize = 1000
        for i in range(0, num_x, blockSize):
            idx = slice(i, min(i + blockSize, num_x))
            x_slice = self.x[idx]

            S_x_sin = np.sin(x_slice[:, np.newaxis] * m_pi_a.flatten())
            S_x_cos = np.cos(x_slice[:, np.newaxis] * m_pi_a.flatten())

            w[idx, :] = base_factor * (S_x_sin @ F_w)
            e_xx[idx, :] = base_factor * (S_x_sin @ F_xx)
            e_yy[idx, :] = base_factor * (S_x_sin @ F_yy)
            e_xy[idx, :] = base_factor * (S_x_cos @ F_xy)

            exx_xi[idx, :] = base_factor * (S_x_sin @ F_exx_xi)
            eyy_xi[idx, :] = base_factor * (S_x_sin @ F_eyy_xi)
            exy_xi[idx, :] = base_factor * (S_x_cos @ F_exy_xi)

            exx_eta[idx, :] = base_factor * (S_x_sin @ F_exx_eta)
            eyy_eta[idx, :] = base_factor * (S_x_sin @ F_eyy_eta)
            exy_eta[idx, :] = base_factor * (S_x_cos @ F_exy_eta)

        eps = np.sqrt(e_xx**2 + e_yy**2 + e_xy**2)
        
        # Mask to prevent division by zero at locations where strain is zero
        mask = eps > 0
        eps_xi = np.zeros_like(eps)
        eps_eta = np.zeros_like(eps)

        eps_xi[mask] = (e_xx[mask] * exx_xi[mask] + 
                        e_yy[mask] * eyy_xi[mask] + 
                        e_xy[mask] * exy_xi[mask]) / eps[mask]

        eps_eta[mask] = (e_xx[mask] * exx_eta[mask] + 
                         e_yy[mask] * eyy_eta[mask] + 
                         e_xy[mask] * exy_eta[mask]) / eps[mask]

        return w, e_xx, e_yy, e_xy, eps, eps_xi, eps_eta


    def strain_xx_xi(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = -(self.t / 2) * (4 * self.P) / \
            (np.pi * self.a**4 * self.b * self.D)

        numerator_load = (m**3 * np.cos(m * np.pi * self.xi / self.a)
                        ) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (sin_x @ C @ sin_y.T)
    
    def strain_xx_eta(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = -(self.t / 2) * (4 * self.P) / \
            (np.pi * self.a**4 * self.b * self.D)

        numerator_load = (m**3 * np.sin(m * np.pi * self.xi / self.a)
                        ) * np.cos(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_yy_eta(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = (self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a * self.b**3 * self.D)

        term_m = m * np.pi / self.a
        term_n = n * np.pi / self.b

        numerator_load = (n**2 * np.sin(term_m * self.xi)) * \
                         (term_n * np.cos(term_n * self.eta))
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_yy_xi(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = (self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a * self.b**3 * self.D)

        term_m = m * np.pi / self.a
        term_n = n * np.pi / self.b

        numerator_load = (n**2 * (term_m * np.cos(term_m * self.xi))) * \
                         np.sin(term_n * self.eta)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_xy_eta(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = -(self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a**2 * self.b**2 * self.D)

        term_m = m * np.pi / self.a
        term_n = n * np.pi / self.b

        numerator_load = np.sin(term_m * self.xi) * \
                         (term_n * np.cos(term_n * self.eta))
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (numerator_load / denominator) * (m * n)

        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (cos_x @ C @ cos_y.T)

    def strain_xy_xi(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = -(self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a**2 * self.b**2 * self.D)

        term_m = m * np.pi / self.a
        term_n = n * np.pi / self.b

        numerator_load = (term_m * np.cos(term_m * self.xi)) * \
                         np.sin(term_n * self.eta)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (numerator_load / denominator) * (m * n)

        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (cos_x @ C @ cos_y.T)