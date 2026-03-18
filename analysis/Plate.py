import numpy as np
import pandas as pd


class Plate:
    """
    Timoshenko/Kirchhoff plate analyzer using the Navier solution.
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
            df.index = pd.Series(np.arange(1, len(df) + 1), name="Element ID")
            self._dataframe = df
        return self._dataframe

    @property
    def dataframe_centroid(self):
        """
        Provides plate analysis results as a pandas DataFrame with values
        calculated directly at the centroid of each grid cell.

        This is achieved by creating a new mesh of centroid coordinates and
        re-running the plate state calculation for that mesh.

        The DataFrame is cached and recomputed only when plate parameters change.
        """
        if self._dataframe_centroid is None:
            original_x, original_y = self.x, self.y
            try:
                # 1. Calculate a new mesh composed of the centroid coordinates
                centroid_x_full = (original_x[:-1] + original_x[1:]) / 2.0
                centroid_y_full = (original_y[:-1] + original_y[1:]) / 2.0

                # Filter for bottom-left quarter of the plate
                centroid_x = centroid_x_full[centroid_x_full <= self._a / 2]
                centroid_y = centroid_y_full[centroid_y_full <= self._b / 2]

                # 2. Temporarily assign this new mesh to the plate instance
                self.x = centroid_x
                self.y = centroid_y

                # 3. Calculate the plate state for the centroid mesh
                w_c, exx_c, eyy_c, exy_c = self.calculate_plate_state()
                eps_c, eps_xi_c, eps_eta_c = self.calculate_eps_variants()

                # 4. Create the DataFrame
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
                # 5. IMPORTANT: Restore the original nodal mesh
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
        m = np.arange(1, self.N + 1)[:, np.newaxis]  # shape (N, 1)
        n = np.arange(1, self.N + 1)[np.newaxis, :]  # shape (1, N)

        scalar_factor = (4 * self.P) / (np.pi**4 * self.a * self.b * self.D)

        numerator_load = np.sin(m * np.pi * self.xi /
                                self.a) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] *
                       (m.flatten() * np.pi / self.a))  # shape (Nx, N)
        sin_y = np.sin(self.y[:, np.newaxis] *
                       (n.flatten() * np.pi / self.b))  # shape (Ny, N)

        return scalar_factor * (sin_x @ C @ sin_y.T)

    def strain_xx(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = (self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a**3 * self.b * self.D)

        numerator_load = (m**2 * np.sin(m * np.pi * self.xi / self.a)
                          ) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = numerator_load / denominator

        sin_x = np.sin(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        sin_y = np.sin(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (sin_x @ C @ sin_y.T)

    def calculate_eps_variants(self):
        """
        Calculates 'eps', 'eps_xi', and 'eps_eta' in a single pass to optimize
        computation by reusing common terms.

        Returns
        -------
        tuple
            A tuple containing the (eps, eps_xi, eps_eta) grids.
        """
        x = self.x
        y = self.y

        scalar_factor = (self.t / 2) * (4 * self.P) / \
            (np.pi**4 * self.a * self.b * self.D)


        eps_yx = np.zeros((len(y), len(x)))
        eps_xi_yx = np.zeros((len(y), len(x)))
        eps_eta_yx = np.zeros((len(y), len(x)))

        m_range = np.arange(1, self.N + 1)
        n_range = np.arange(1, self.N + 1)

        # Common terms for all calculations
        m_pi_a = m_range * np.pi / self.a
        n_pi_b = n_range * np.pi / self.b

        sin_mx_sq = np.sin(np.outer(m_pi_a, x))**2
        cos_mx_sq = np.cos(np.outer(m_pi_a, x))**2
        sin_ny_sq = np.sin(np.outer(n_pi_b, y))**2
        cos_ny_sq = np.cos(np.outer(n_pi_b, y))**2

        m2_a2 = (m_range/self.a)**2
        n2_b2 = (n_range/self.b)**2
        m4_a4 = m2_a2**2
        n4_b4 = n2_b2**2

        # --- Calculate Coefficients for eps, eps_xi, and eps_eta ---
        denominator = (m2_a2[:, None] + n2_b2[None, :])**2
        
        sin_m_xi = np.sin(m_pi_a * self.xi)
        cos_m_xi = np.cos(m_pi_a * self.xi)
        sin_n_eta = np.sin(n_pi_b * self.eta)
        cos_n_eta = np.cos(n_pi_b * self.eta)

        coeffs_eps = (sin_m_xi[:, None] * sin_n_eta[None, :]) / denominator
        
        coeffs_eps_xi = (cos_m_xi[:, None] * sin_n_eta[None, :]) / denominator
        coeffs_eps_xi *= m_pi_a[:, None]

        coeffs_eps_eta = (sin_m_xi[:, None] * cos_n_eta[None, :]) / denominator
        coeffs_eps_eta *= n_pi_b[None, :]

        # --- Main calculation loop (chunked for memory efficiency) ---
        chunk_size = 100
        for i in range(0, len(m_range), chunk_size):
            m_slice = slice(i, i + chunk_size)
            m2_sub = m2_a2[m_slice]
            m4_sub = m4_a4[m_slice]
            sin_mx_sub = sin_mx_sq[m_slice]
            cos_mx_sub = cos_mx_sq[m_slice]

            for j in range(0, len(n_range), chunk_size):
                n_slice = slice(j, j + chunk_size)
                n2_sub = n2_b2[n_slice]
                n4_sub = n4_b4[n_slice]
                sin_ny_sub = sin_ny_sq[n_slice]
                cos_ny_sub = cos_ny_sq[n_slice]

                # Common square root term
                term_A_chunk = (m4_sub[:, None] + n4_sub[None, :])[:, :, None, None] * sin_ny_sub[None, :, :, None] * sin_mx_sub[:, None, None, :]
                term_B_chunk = (m2_sub[:, None] * n2_sub[None, :])[:, :, None, None] * cos_ny_sub[None, :, :, None] * cos_mx_sub[:, None, None, :]
                sqrt_term = np.sqrt(np.pi**4 * (term_A_chunk + term_B_chunk))

                # Accumulate results for each variant
                coeffs_eps_sub = coeffs_eps[m_slice, n_slice]
                eps_yx += np.sum(coeffs_eps_sub[:, :, None, None] * sqrt_term, axis=(0, 1))

                coeffs_eps_xi_sub = coeffs_eps_xi[m_slice, n_slice]
                eps_xi_yx += np.sum(coeffs_eps_xi_sub[:, :, None, None] * sqrt_term, axis=(0, 1))

                coeffs_eps_eta_sub = coeffs_eps_eta[m_slice, n_slice]
                eps_eta_yx += np.sum(coeffs_eps_eta_sub[:, :, None, None] * sqrt_term, axis=(0, 1))

        # Transpose to match (Nx, Ny) convention of this class
        return scalar_factor * eps_yx.T, scalar_factor * eps_xi_yx.T, scalar_factor * eps_eta_yx.T

    def strain_yy(self):
        m = np.arange(1, self.N + 1)[:, np.newaxis]
        n = np.arange(1, self.N + 1)[np.newaxis, :]

        scalar_factor = (self.t / 2) * (4 * self.P) / \
            (np.pi**2 * self.a * self.b**3 * self.D)

        numerator_load = (n**2 * np.sin(m * np.pi * self.xi / self.a)
                          ) * np.sin(n * np.pi * self.eta / self.b)
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

        numerator_load = np.sin(m * np.pi * self.xi /
                                self.a) * np.sin(n * np.pi * self.eta / self.b)
        denominator = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (numerator_load / denominator) * (m * n)

        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return scalar_factor * (cos_x @ C @ cos_y.T)

    def compute_strain_xy_segment(self, m_range, n_range):
        if len(m_range) == 0 or len(n_range) == 0:
            return 0

        m = np.array(m_range)[:, np.newaxis]
        n = np.array(n_range)[np.newaxis, :]

        load_term = np.sin(m * np.pi * self.xi / self.a) * \
            np.sin(n * np.pi * self.eta / self.b)
        den = ((m / self.a)**2 + (n / self.b)**2)**2
        C = (load_term / den) * (m * n)

        cos_x = np.cos(self.x[:, np.newaxis] * (m.flatten() * np.pi / self.a))
        cos_y = np.cos(self.y[:, np.newaxis] * (n.flatten() * np.pi / self.b))

        return cos_x @ C @ cos_y.T

    def strain_xy_delta(self, N_start, N_end):
        if N_start >= N_end:
            return np.zeros((len(self.x), len(self.y)))

        scalar_factor = -self.t * (4 * self.P) / \
            (np.pi**2 * self.a**2 * self.b**2 * self.D)

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

        load_term = np.sin(m * np.pi * self.xi / self.a) * \
            np.sin(n * np.pi * self.eta / self.b)
        den = ((m / self.a)**2 + (n / self.b)**2)**2
        C_base = load_term / den

        C_w = C_base
        C_xx = C_base * (m**2) * (np.pi**2 / self.a**2) * (self.t / 2)
        C_yy = C_base * (n**2) * (np.pi**2 / self.b**2) * (self.t / 2)
        C_xy = C_base * (m * n) * (np.pi**2 /
                                   (self.a * self.b)) * (-self.t / 2)

        S_y_sin = np.sin(n.flatten() * np.pi / self.b *
                         self.y[:, np.newaxis]).T
        S_y_cos = np.cos(n.flatten() * np.pi / self.b *
                         self.y[:, np.newaxis]).T

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

            S_x_sin_block = np.sin(
                x_slice[:, np.newaxis] * (m.flatten() * np.pi / self.a))
            S_x_cos_block = np.cos(
                x_slice[:, np.newaxis] * (m.flatten() * np.pi / self.a))

            w[idx, :] = base_factor * (S_x_sin_block @ F_w)
            e_xx[idx, :] = base_factor * (S_x_sin_block @ F_xx)
            e_yy[idx, :] = base_factor * (S_x_sin_block @ F_yy)
            e_xy[idx, :] = base_factor * (S_x_cos_block @ F_xy)

        return w, e_xx, e_yy, e_xy
    
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
