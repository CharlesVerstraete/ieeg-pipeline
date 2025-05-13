#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Signal processing helper functions
"""

# Import libraries
from preprocessing.config import *
import meegkit as mk 
from scipy.signal import welch
from joblib import Parallel, delayed

def compute_line_noise_band(f, noise_fline, bandwidth=1.5, win=10):
    """
    Compute the line noise and the signal band
    """ 
    noise_band = (f >= noise_fline-bandwidth) & (f <= noise_fline+bandwidth)
    signal_band = (f >= noise_fline-win) & (f <= noise_fline+win)
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
    print(f"Removing noise at {noise_fline} Hz, {data.shape[0]} channels, nfft = {nfft}")
    to_clean = deepcopy(data)
    clean, _ = mk.dss.dss_line(to_clean.T, noise_fline, sfreq, nremove=1, nfft=nfft)
    del to_clean
    gc.collect()
    return clean.T

def get_batch_data(data, snr, bad_idx, batch_size):
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


def clean_line_noise(data, noise_fline, sfreq, snr, noise_band, signal_band, nfft, batch_size, snr_threshold=0.8, iter_max=10):
    """
    Clean the line noise from the data
    """
    iteration = 0
    bad_idx = np.where((snr < snr_threshold))[0]
    data_ = deepcopy(data)
    while (len(bad_idx) > 5) & (iteration < iter_max) :
        print(f"Iteration {iteration}, {len(bad_idx)} bad channels, bad snr avr = {np.mean(snr[bad_idx]):.2f}")
        batch_data, batch_indices, n_batches = get_batch_data(data_, snr, bad_idx, batch_size)
        clean = Parallel(n_jobs=-1)(
            delayed(process_channel_batch)(
                batch_data[i], noise_fline, sfreq, nfft
            ) for i in range(n_batches)
        )
        for i, idx_batch in enumerate(batch_indices):
            data_[idx_batch, :] = clean[i]
        _ , pxx = compute_psd(data_, sfreq, nperseg = 4*sfreq, fmin=2, fmax=200)
        snr = compute_snr(pxx, signal_band, noise_band)
        bad_idx = np.where((snr < snr_threshold))[0]
        iteration += 1
    return data_


def compute_psd(data, sfreq, nperseg, min=8, fmin=None, fmax=None):
    """
    Compute the power spectral density
    """
    n_channels, _ = data.shape
    if n_channels <= min:
        return welch(data, fs=sfreq, nperseg=nperseg)

    channel_batches = np.array_split(np.arange(n_channels), min)

    def _process_psd_batch(ch_idx):
        batch_data = data[ch_idx]
        f, p = welch(batch_data, fs=sfreq, nperseg=nperseg)
        return f, p

    results = Parallel(n_jobs=-1)(
        delayed(_process_psd_batch)(batch) for batch in channel_batches
    )
    
    f = results[0][0]
    pxx = np.vstack([r[1] for r in results])

    if fmin is not None and fmax is not None:
        mask = (f >= fmin) & (f <= fmax)
        f = f[mask]
        pxx = pxx[:, mask]
    
    return f, pxx


def get_clean_electrode(electrode, ch_names, subject):
    """
    Get the electrodes that are in gm and in signal
    """
    clean_electrode = electrode[electrode["anode"].isin(ch_names) & electrode["cathode"].isin(ch_names)]
    if len(BAD_CHANNELS[subject]) > 0:
        bad = BAD_CHANNELS[subject]
        anode = clean_electrode["anode"]
        cathode = clean_electrode["cathode"]
        idx = ~anode.isin(bad) & ~cathode.isin(bad)
        clean_electrode = clean_electrode[idx]
    return clean_electrode


def check_lenght(raw, sfreq, start_idx, end_idx, epoch_lenght):
    """
    Check if the raw data is long enough to create the epochs
    """
    raw_size = len(raw.times)
    end_delta = raw_size - end_idx
    start_delta = start_idx
    pad_added = False
    n_miss_end = 0 
    n_miss_start = 0 
    if end_delta < epoch_lenght*sfreq :
        n_miss_end = (epoch_lenght*sfreq - end_delta).astype(int) + 1
        raw = add_padding(n_miss_end, raw, "end")
        pad_added = True
    if start_delta < epoch_lenght*sfreq :
        n_miss_start = (epoch_lenght*sfreq - start_delta).astype(int) + 1
        raw = add_padding(n_miss_start, raw, "start")
        pad_added = True
    return raw, pad_added, n_miss_start, n_miss_end

def add_padding(n_miss, raw, where):
    """
    Add padding to the raw data
    """
    orig_data = raw.get_data()  
    n_ch, _ = orig_data.shape
    padding_data = np.empty((n_ch, n_miss))
    for channel in range(n_ch):
        channel_data = orig_data[channel]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        padding_data[channel] = np.random.normal(mean, std, n_miss)
    
    if where == "end":
        new_data = np.concatenate((orig_data, padding_data), axis=1)
    elif where == "start":
        new_data = np.concatenate((padding_data, orig_data), axis=1)

    new_info = raw.info.copy()
    del orig_data, padding_data
    return mne.io.RawArray(new_data, new_info)