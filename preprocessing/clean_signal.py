#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

"""
Filter, bipolarize and denoise the iEEG signals
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import get_fileslist
from preprocessing.utils.signal_helper import *
import mne
import meegkit as mk 
from copy import deepcopy
import gc
from scipy.signal import welch

import matplotlib
matplotlib.use('Qt5Agg')


subject = SUBJECTS[0]

electrode_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "anat", f"sub-{int(subject):03}_electrodes-bipolar.csv")
electrode = pd.read_csv(electrode_path)
events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
events = pd.read_csv(events_path, sep = "\t")
n_run = events["run"].max().astype(int)

run = 1
raw_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_run-{run:02}-raw.fif")
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
clean_electrode = electrode[electrode["anode"].isin(raw.ch_names) & electrode["cathode"].isin(raw.ch_names)]

if len(BAD_CHANNELS[subject]) > 0:
    bad = BAD_CHANNELS[subject]
    anode = clean_electrode["anode"].str.lower()
    cathode = clean_electrode["cathode"].str.lower()
    idx = ~anode.isin(bad) & ~cathode.isin(bad)
    clean_electrode = clean_electrode[idx]

clean_electrode = clean_electrode[clean_electrode["is_inmask"]]
sfreq = raw.info["sfreq"]

psd_raw = raw.compute_psd(fmin=0, fmax=512, picks="seeg", n_jobs=-1, n_fft=int(2*sfreq))
psd_raw.plot()

anodes = clean_electrode["anode"].values.astype(str).tolist()
cathodes = clean_electrode["cathode"].values.astype(str).tolist()
bipolar_raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, drop_refs= True)
del raw
gc.collect()
bipolar_name = clean_electrode["name"].values.astype(str).tolist()

bipolar_raw.filter(l_freq=HIGHPASS, h_freq=LOWPASS, n_jobs=-1)

psd_bipolar = bipolar_raw.compute_psd(fmin=2, fmax=512, picks=bipolar_name, n_jobs=-1, n_fft=int(2*sfreq))
psd_bipolar.plot()

data = bipolar_raw.get_data(picks=bipolar_name).copy()

f, pxx = compute_psd(data, sfreq, nperseg=4*sfreq, fmin=2, fmax=200)

noise_band, signal_band = compute_line_noise_band(f, LINENOISE, bandwidth=2, win=10)
snr = compute_snr(pxx, signal_band, noise_band)

test_data = clean_line_noise(data, LINENOISE, sfreq, snr, noise_band, signal_band, snr_threshold=0.92, iter_max=10)


f, pxx_clean = compute_psd(test_data, sfreq, nperseg=4*sfreq, fmin=2, fmax=200)


fig, axs = plt.subplots(3, 1, figsize=(21, 12))
axs[0].plot(f, 10 * np.log10(pxx).T, lw = 0.2, color = "grey")
axs[0].set_title("PSD before cleaning")
axs[1].plot(f, 10 * np.log10(pxx_clean).T, lw = 0.2, color = "grey")
axs[1].set_title("PSD after cleaning")
axs[2].plot(f, ((10 * np.log10(pxx)) - (10 * np.log10(pxx_clean))).T, lw = 0.2, color = "firebrick")
axs[2].set_title("Difference between before and after cleaning")
plt.tight_layout()
plt.show()


