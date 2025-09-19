# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour extraire et préparer les caractéristiques pour l'analyse
"""

from analysis.decoding.config import *
# from config import *
from joblib import Parallel, delayed

class Features:
    def __init__(self, power_data, phase_data, subject):
        self.power_data = power_data
        self.phase_data = phase_data
        self.subject = subject
        self.n_epochs = power_data.shape[0]
        self.n_channels = power_data.shape[1]
        self.all_features = [(fr, ch) for fr in range(n_frband) for ch in range(self.n_channels)]
        self.power_bands = None
        self.phase_bands = None
        self.phase_sin_bands = None
        self.phase_cos_bands = None

    def baseline_signal(self) :
        """Baseline the signal for power data"""
        baseline_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "timefreq", f"sub-{int(self.subject):03}_tfr-baseline.npy")
        baseline = np.load(baseline_path)
        self.power_data -= baseline
        del baseline
        gc.collect()

    def _compute_power_band_channel(self, i, ch):
        """Compute power for a specific frequency band and channel"""
        return np.mean(self.power_data[:, ch, band_indices[i]:band_indices[i+1], :], axis=1)

    def extract_power_bands(self, n_jobs=-1):
        """Extract power bands with parallel processing across channels and frequency bands"""

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_power_band_channel)(*task) for task in self.all_features
        )
        
        power_bands = np.zeros((self.n_epochs, self.n_channels, n_frband, n_times_decimed), dtype=np.float32)
        
        for idx, (fr, ch) in enumerate(self.all_features):
            power_bands[:, ch, fr, :] = results[idx]
            
        self.power_bands = power_bands
    
    def _compute_phase_band_channel(self, i, ch):
        """Compute phase features for a specific frequency band and channel"""
        phase_band = np.array(self.phase_data[:, ch, band_indices[i]:band_indices[i+1], :], dtype=np.float32)
        
        complex_mean = np.exp(1j * phase_band).mean(axis=1)
        mean_phase = np.angle(complex_mean)
        
        return {
            'phase': mean_phase,
            'sin': np.sin(mean_phase),
            'cos': np.cos(mean_phase)
        }
    
    def extract_phase_bands(self, n_jobs=-1):
        """Extract phase bands with parallel processing across channels and frequency bands"""

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_phase_band_channel)(*task) for task in self.all_features
        )
        
        phase_bands = np.zeros((self.n_epochs, self.n_channels, n_frband, n_times_decimed), dtype=np.float32)
        phase_sin_bands = np.zeros((self.n_epochs, self.n_channels, n_frband, n_times_decimed), dtype=np.float32)
        phase_cos_bands = np.zeros((self.n_epochs, self.n_channels, n_frband, n_times_decimed), dtype=np.float32)
        
        for idx, (fr, ch) in enumerate(self.all_features):
            result = results[idx]
            phase_bands[:, ch, fr, :] = result['phase']
            phase_sin_bands[:, ch, fr, :] = result['sin']
            phase_cos_bands[:, ch, fr, :] = result['cos']
        
        self.phase_bands = phase_bands
        self.phase_sin_bands = phase_sin_bands
        self.phase_cos_bands = phase_cos_bands