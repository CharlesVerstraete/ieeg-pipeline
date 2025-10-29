# -*- coding: utf-8 -*-
# -*- python 3.9.6 -*-

"""
Module pour extraire et préparer les caractéristiques pour l'analyse
"""

from analysis.decoding.config import *
from analysis.decoding.config import *
from analysis.decoding.core_classes.loader import iEEGDataLoader
from joblib import Parallel, delayed

class Features:
    def __init__(self, subject, model=""):
        self.loader = iEEGDataLoader(subject)
        self.subject = subject
        self.n_epochs = self.loader.n_epochs
        self.n_channels = self.loader.n_channels
        self.all_features = [(fr, ch) for fr in range(n_frband) for ch in range(self.n_channels)]
        self.beh, self.event_to_keep, self.events_df = self.loader.load_behavior(model)
        self.events_dict = self.loader.get_events_dict(self.beh, self.events_df)
        self.power_data = None
        self.baseline = None
        self.power_data_baselined = None
        self.phase_data = None
        self.power_bands = None
        self.phase_bands = None
        self.phase_sin_bands = None
        self.phase_cos_bands = None

    def load_data(self, phase=True, path=None):
        """Load power and/or phase data"""
        self.power_data = self.loader.load_power(path)
        self.baseline = self.loader.load_baseline()
        if path is not None:
            self.n_epochs = self.power_data.shape[0]
            self.baseline = self.baseline[self.event_to_keep, :, :, :]
        if phase:
            self.phase_data = self.loader.load_phase()

    def prepare_data(self):
        self.power_data = self.power_data[self.event_to_keep, :, :, :]
        self.baseline = self.baseline[self.event_to_keep, :, :, :]
        self.n_epochs = len(self.event_to_keep)
        if self.phase_data is not None:
            self.phase_data = self.phase_data[self.event_to_keep, :, :, :]

    def _realign_data(self, event_name, tmin=-1, tmax=1, phase=False):
        """Realign data to a specific event"""
        event_samples = self.events_dict[event_name]
        start_offset = int(tmin * sr_decimated)
        end_offset = int(tmax * sr_decimated)
        win_len = end_offset - start_offset
        realign_power_data = np.zeros((self.n_epochs, self.n_channels, n_freqs, win_len), dtype=np.float16)
        padded_list = []
        if phase:
            realign_phase_data = np.zeros((self.n_epochs, self.n_channels, n_freqs, win_len), dtype=np.float16)
        else:
            realign_phase_data = None
        for i in range(self.n_epochs):
            start_idx = event_samples[i] + start_offset
            end_idx = event_samples[i] + end_offset
            if end_idx > n_times_decimed:
                syntetic_padding = end_idx - n_times_decimed
                mu = self.baseline[i].astype(np.float32)
                sigma = np.std(self.power_data[i].astype(np.float32), axis=-1, keepdims=True)
                syntetic_data = np.random.normal(mu, sigma, (self.n_channels, n_freqs, syntetic_padding))
                realign_power_data[i, :, :, :win_len-syntetic_padding] = self.power_data[i, :, :, start_idx:end_idx]
                realign_power_data[i, :, :, win_len-syntetic_padding:] = syntetic_data
                padded_list.append(syntetic_padding)
            else :
                realign_power_data[i, :, :, :] = self.power_data[i, :, :, start_idx:end_idx]
                padded_list.append(0)
            if phase:
                realign_phase_data[i, :, :, :] = self.phase_data[i, :, :, start_idx:end_idx]
        self.events_dict[f'padded_{event_name}'] = np.array(padded_list)
        return realign_power_data, realign_phase_data
    
    def export_realign_data(self, event_name, tmin=-1, tmax=1, phase=False):
        """Export realigned data to .npy files"""
        realign_power_data, realign_phase_data = self._realign_data(event_name, tmin, tmax, phase)
        power_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "aligned", f"sub-{int(self.subject):03}_tfr-realign-{event_name}_{tmin}-{tmax}_power.npy")
        np.save(power_path, realign_power_data)
        if phase:
            phase_path = os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "aligned", f"sub-{int(self.subject):03}_tfr-realign-{event_name}_{tmin}-{tmax}_phase.npy")
            np.save(phase_path, realign_phase_data)
        metadata = {
            'sfreq': sr_decimated,
            'n_epochs': self.n_epochs,
            'keep_events': self.event_to_keep.to_list(),
            'n_channels': self.n_channels,
            'ch_names': self.loader.ch_names,
            'n_freqs' : n_freqs,
            'samples_event': self.events_dict[event_name].tolist(),
            'missing_event': self.events_dict[f'is_missing_{event_name}'].tolist(),
            'padded_event': self.events_dict[f'padded_{event_name}'].tolist(),
            'tmin': tmin,
            'tmax': tmax
        }
        metadata_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(self.subject):03}", "preprocessed", "aligned", f"sub-{int(self.subject):03}_tfr-realign-{event_name}-{tmin}-{tmax}_metadata.json"))
        with open(metadata_path, 'w', encoding='utf-8') as f: 
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        del realign_power_data, realign_phase_data
        gc.collect()


    def baseline_signal(self) :
        """Baseline the signal for power data"""
        self.power_data_baselined = self.power_data - self.baseline

    def _compute_power_band_channel(self, i, ch, baselined=False):
        """Compute power for a specific frequency band and channel"""
        res = None
        if baselined:
            res = np.mean(self.power_data_baselined[:, ch, band_indices[i]:band_indices[i+1], :], axis=1)
        else:
            res = np.mean(self.power_data[:, ch, band_indices[i]:band_indices[i+1], :], axis=1)
        return res

    def extract_power_bands(self, n_jobs=-1, baselined=False):
        """Extract power bands with parallel processing across channels and frequency bands"""

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_power_band_channel)(*task, baselined=baselined) for task in self.all_features
        )

        power_bands = np.zeros((self.n_epochs, self.n_channels, n_frband, self.power_data.shape[-1]), dtype=np.float32)

        for idx, (fr, ch) in enumerate(self.all_features):
            power_bands[:, ch, fr, :] = results[idx]
            
        self.power_bands = power_bands

        del results, power_bands
        gc.collect()
    
    def _compute_phase_band_channel(self, i, ch):
        """Compute phase features for a specific frequency band and channel"""
        phase_band = np.array(self.phase_data[:, ch, band_indices[i]:band_indices[i+1], :], dtype=np.float32)
        
        complex_mean = np.exp(1j * phase_band).mean(axis=1)
        mean_phase = np.angle(complex_mean)
        
        del phase_band, complex_mean
        gc.collect()

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
    
        del results, phase_bands, phase_sin_bands, phase_cos_bands
        gc.collect()

