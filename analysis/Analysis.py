import numpy as np
from OptimizedPlate import OptimizedPlate
import pandas as pd
import os

class Analysis:
    def __init__(self, source, tol=1e-7):
        if isinstance(source, OptimizedPlate):
            self.p = source
            self.tol = tol
    
            self.fs_x = 1.0 / self.p.space_res_x
            self.fs_y = 1.0 / self.p.space_res_y
    
            self.kx = np.linspace(-self.fs_x / 2, self.fs_x / 2, len(self.p.x))
            self.ky = np.linspace(-self.fs_y / 2, self.fs_y / 2, len(self.p.y))
    
            self.find_convergence()
    
            self.w, self.exx, self.eyy, self.exy = self.p.calculate_plate_state()
        elif isinstance(source, str) and os.path.exists(source):
            self.p = None # No OptimizedPlate object
            self.tol = tol
            
            df = pd.read_parquet(source)

            # Reconstruct 2D arrays. The pivot creates (Y, X) shaped arrays, so we transpose.
            self.exx = df.pivot(index='Y', columns='X', values='EXX').to_numpy().T
            self.eyy = df.pivot(index='Y', columns='X', values='EYY').to_numpy().T
            self.exy = df.pivot(index='Y', columns='X', values='EXY').to_numpy().T
            self.w = df.pivot(index='Y', columns='X', values='Z').to_numpy().T

            x_coords = sorted(df['X'].unique())
            y_coords = sorted(df['Y'].unique())

            # Estimate space_res. This assumes uniform grid.
            space_res_x = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
            space_res_y = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0

            self.fs_x = 1.0 / space_res_x
            self.fs_y = 1.0 / space_res_y

            self.kx = np.linspace(-self.fs_x / 2, self.fs_x / 2, len(x_coords))
            self.ky = np.linspace(-self.fs_y / 2, self.fs_y / 2, len(y_coords))
        else:
            raise ValueError("source must be a OptimizedPlate object or a valid Parquet file path")

        self.ft_exx = np.fft.fftshift(np.fft.fft2(self.exx.T))
        self.ft_eyy = np.fft.fftshift(np.fft.fft2(self.eyy.T))
        self.ft_exy = np.fft.fftshift(np.fft.fft2(self.exy.T))
        self.ft_w = np.fft.fftshift(np.fft.fft2(self.w.T))

    @property
    def N(self):
        if self.p is None:
            raise AttributeError("Cannot get N when Analysis is initialized from a Parquet file.")
        return self.p.N

    @N.setter
    def N(self, value):
        if self.p is None:
            raise AttributeError("Cannot set N when Analysis is initialized from a Parquet file.")
        self.p.N = value

    @property
    def xi(self):
        if self.p is None:
            raise AttributeError("Cannot get xi when Analysis is initialized from a Parquet file.")
        return self.p.xi

    @xi.setter
    def xi(self, value):
        if self.p is None:
            raise AttributeError("Cannot set xi when Analysis is initialized from a Parquet file.")
        self.p.xi = value

    @property
    def eta(self):
        if self.p is None:
            raise AttributeError("Cannot get eta when Analysis is initialized from a Parquet file.")
        return self.p.eta

    @eta.setter
    def eta(self, value):
        if self.p is None:
            raise AttributeError("Cannot set eta when Analysis is initialized from a Parquet file.")
        self.p.eta = value

    def find_convergence(self):
        if self.p is None:
            print("Convergence analysis is not available when initialized from a Parquet file.")
            return

        self.p.N = 8000
        return
        
        # max_diff = np.inf
        # n_current = 1000
        # n_step = 250

        # self.p.N = n_current

        # X, Y = np.meshgrid(self.p.x, self.p.y, indexing='ij')
        # dist_from_load = np.sqrt((X - self.p.xi)**2 + (Y - self.p.eta)**2)
        # mask = dist_from_load > 0.05

        # while max_diff > self.tol:
        #     n_next = n_current + n_step
        #     delta = self.p.strain_xy_delta(n_current, n_next)
        #     max_diff = np.max(np.abs(delta[mask]))

        #     n_current = n_next
        #     if n_current > 10000:
        #         print('Convergence not reached within 10000 terms.')
        #         break

        # self.p.N = n_current

    def update(self):
        if self.p is None:
            raise AttributeError("Cannot update when Analysis is initialized from a Parquet file.")
        self.find_convergence()
        self.w, self.exx, self.eyy, self.exy = self.p.calculate_plate_state()
        self.ft_exx = np.fft.fftshift(np.fft.fft2(self.exx.T))
        self.ft_eyy = np.fft.fftshift(np.fft.fft2(self.eyy.T))
        self.ft_exy = np.fft.fftshift(np.fft.fft2(self.exy.T))
        self.ft_w = np.fft.fftshift(np.fft.fft2(self.w.T))

    def _calculate_cutoff(self, ft_field, label, threshold=0.99):
        psd = np.abs(ft_field)**2
        
        energy_x = np.sum(psd, axis=0)
        cum_x = np.cumsum(energy_x) / np.sum(energy_x)
        fs_max_x = np.abs(self.kx[np.argmax(cum_x >= threshold)])

        energy_y = np.sum(psd, axis=1)
        cum_y = np.cumsum(energy_y) / np.sum(energy_y)
        fs_max_y = np.abs(self.ky[np.argmax(cum_y >= threshold)])

        print(f"{'='*80}\nRISULTATI ANALISI SPETTRALE {label}\n{'-'*80}")
        for direction, freq in zip(['X', 'Y'], [fs_max_x, fs_max_y]):
            print(f"Direzione {direction}\nNumero d'onda critico ({int(threshold*100)}% energia): {freq:.2f} 1/m")
            print(f"Grandezza mesh massima: {1 / (2 * freq) * 1000:.2f} mm")
        
        return fs_max_x, fs_max_y

    def find_sampling_freq_exx(self, threshold=0.99):
        return self._calculate_cutoff(self.ft_exx, "e_xx", threshold)

    def find_sampling_freq_eyy(self, threshold=0.99):
        return self._calculate_cutoff(self.ft_eyy, "e_yy", threshold)

    def find_sampling_freq_exy(self, threshold=0.99):
        return self._calculate_cutoff(self.ft_exy, "e_xy", threshold)