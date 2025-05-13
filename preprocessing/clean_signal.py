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
from preprocessing.utils.plot_helper import *

# for subject in [9, 12, 14, 16, 19, 20, 23, 25, 28]:
# for subject in [25, 28]:
subject = 3
print(f"Processing subject {subject}")
electrode_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "anat", f"sub-{int(subject):03}_electrodes-bipolar.csv")
electrode = pd.read_csv(electrode_path)
events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
events = pd.read_csv(events_path, sep = "\t")
n_run = events["run"].max().astype(int)
# for run in range(1, n_run + 1):
run = 1
raw_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_run-{run:02}-raw.fif")
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
clean_electrode = get_clean_electrode(electrode, raw.ch_names, subject)
sfreq = int(raw.info["sfreq"])
anodes = clean_electrode["anode"].values.astype(str).tolist()
cathodes = clean_electrode["cathode"].values.astype(str).tolist()
bipolar_raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, drop_refs= True, verbose=False)
del raw
gc.collect()
bipolar_name = clean_electrode["name"].values.astype(str).tolist()
bipolar_raw.filter(l_freq=HIGHPASS, h_freq=LOWPASS, n_jobs=-1, verbose=False)

data = bipolar_raw.get_data(picks=bipolar_name).copy()
f, pxx = compute_psd(data, sfreq, nperseg=4*sfreq, fmin=2, fmax=200)
noise_band, signal_band = compute_line_noise_band(f, LINENOISE, bandwidth=2, win=10)
snr = compute_snr(pxx, signal_band, noise_band)
clean = clean_line_noise(data, LINENOISE, sfreq, snr, noise_band, signal_band, int(sfreq*2), 10, snr_threshold=0.8, iter_max=10)
f, pxx_clean = compute_psd(clean, sfreq, nperseg=4*sfreq, fmin=2, fmax=200)
fig_savepath = os.path.join(FIGURES_DIR, "preprocessing", "psd", f"sub-{int(subject):03}_run-{run:02}_psd_clean.pdf")
plot_denoised(f, pxx, pxx_clean, fig_savepath)
bipolar_picks = mne.pick_channels(bipolar_raw.ch_names, bipolar_name) 
bipolar_raw._data[bipolar_picks] = clean
clean_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "filtered", f"sub-{int(subject):03}_run-{run:02}_bipolar-denoised.fif")
bipolar_raw.save(clean_path, overwrite=True, verbose=False)
del bipolar_raw, data, clean, pxx, pxx_clean, f, noise_band, signal_band, snr
gc.collect()
print(f"sub-{int(subject):03} run-{run:02} done")

############################################################################################################################################################################
############################################################################################################################################################################
# Check bad channels
############################################################################################################################################################################
############################################################################################################################################################################


# for subject in [4, 12, 14]:
subject = 25
electrode_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "anat", f"sub-{int(subject):03}_electrodes-bipolar.csv")
electrode = pd.read_csv(electrode_path)
events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
events = pd.read_csv(events_path, sep = "\t")
n_run = events["run"].max().astype(int)

run = 1
raw_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_run-{run:02}-raw.fif")
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
clean_electrode = get_clean_electrode(electrode, raw.ch_names, subject)
anodes = clean_electrode["anode"].values.astype(str).tolist()
cathodes = clean_electrode["cathode"].values.astype(str).tolist()
bipolar_raw = mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, drop_refs= True, verbose=False)
psd_raw = raw.compute_psd(fmin=0, fmax=256, picks="seeg", n_jobs=-1, n_fft=int(2*sfreq))
psd_raw.plot()
sfreq = raw.info["sfreq"]
psd_raw = bipolar_raw.compute_psd(fmin=0, fmax=256, picks="seeg", n_jobs=-1, n_fft=int(2*sfreq))
psd_raw.plot()
plt.suptitle(f"sub-{int(subject):03} run-{run:02} raw")
plt.show()
del raw, psd_raw, bipolar_raw
gc.collect()

