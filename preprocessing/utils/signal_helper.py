#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Signal processing helper functions
"""

# Import libraries
from preprocessing.config import *
import numpy as np
import mne
import meegkit as mk 
from copy import deepcopy
import gc
from scipy.signal import welch
from joblib import Parallel, delayed

def compute_line_noise_band(f, freq, bandwidth=1.5, win=10):
    """
    Compute the line noise and the signal band
    """ 
    noise_band = (f >= freq-bandwidth) & (f <= freq+bandwidth)
    signal_band = (f >= freq-win) & (f <= freq+win)
    signal_band = signal_band & ~noise_band
    return noise_band, signal_band

def compute_snr(pxx, signal_band, noise_band) : 
    """
    Compute the signal to noise ratio
    """
    return np.mean(pxx[:, signal_band], axis=1) / np.mean(pxx[:, noise_band], axis=1)

def process_channel_batch(data, noise_fline, sfreq, nfft):
    """
    Process a batch of channels to remove line noise
    """
    to_clean = deepcopy(data)
    clean, _ = mk.dss.dss_line(to_clean.T, noise_fline, sfreq, nremove=1, nfft=nfft)
    del to_clean
    gc.collect()
    return clean.T

def get_batch_data(data, bad_idx, snr, batch_size):
    """
    Get the batch data for the bad channels
    """
    bad_snr = snr[bad_idx]
    sorted_indices = np.argsort(bad_snr)
    sorted_bad_idx = np.array(bad_idx)[sorted_indices]

    n_bad = len(bad_idx)
    n_batches = max(1, min(n_bad // batch_size, 8))

    batch_indices = np.array_split(sorted_bad_idx, n_batches)
    batch_data = [data[idx, :] for idx in batch_indices]

    del sorted_indices, sorted_bad_idx, n_bad
    gc.collect()
    return batch_data, batch_indices, n_batches


def clean_line_noise(data, noise_fline, sfreq, snr, noise_band, signal_band, snr_threshold=0.8, iter_max=10):
    """
    Clean the line noise from the data
    """
    iteration = 1
    bad_idx = np.where((snr < snr_threshold))[0]
    data_ = deepcopy(data)
    while (len(bad_idx) > 3) & (iteration < iter_max) :
        batch_data, batch_indices, n_batches = get_batch_data(data_, bad_idx, snr, 8)
        clean = Parallel(n_jobs=8)(
            delayed(process_channel_batch)(
                batch_data[i], LINENOISE, sfreq, sfreq
            ) for i in range(n_batches)
        )
        for i, idx_batch in enumerate(batch_indices):
            data_[idx_batch, :] = clean[i]
        _ , pxx = compute_psd(data_, sfreq, nperseg=4*sfreq, fmin=2, fmax=200)
        snr = compute_snr(pxx, signal_band, noise_band)
        bad_idx = np.where((snr < snr_threshold))[0]
        iteration += 1
    return data_


def compute_psd(data, sfreq, nperseg, n_jobs=8, fmin=None, fmax=None):
    """
    Compute the power spectral density
    """
    n_channels, _ = data.shape
    if n_channels <= n_jobs:
        return welch(data, fs=sfreq, nperseg=nperseg)

    channel_batches = np.array_split(np.arange(n_channels), n_jobs)

    def _process_psd_batch(ch_idx):
        batch_data = data[ch_idx]
        f, p = welch(batch_data, fs=sfreq, nperseg=nperseg)
        return f, p

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_psd_batch)(batch) for batch in channel_batches
    )
    
    f = results[0][0]
    pxx = np.vstack([r[1] for r in results])

    if fmin is not None and fmax is not None:
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        pxx = pxx[:, mask]
    
    return f, pxx


