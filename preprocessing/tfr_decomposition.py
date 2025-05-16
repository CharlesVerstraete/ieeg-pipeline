#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Time-frequency decomposition of the epoched signal
"""

from preprocessing.config import *
from preprocessing.utils.data_helper import *
import json
from tqdm import tqdm


subject = 14

ch_names = epochs.info["ch_names"]
sfreq = epochs.info["sfreq"]

n_epochs = len(epochs)
n_channels = len(ch_names)

power_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-power.npy")
phase_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-phase.npy")

power_complete = np.memmap(power_path, dtype = 'float16',mode = 'w+', shape = (n_epochs, n_channels, n_freqs, n_times_decimed))

# for subject in SUBJECTS:
# metadata_path =  os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "epochs", f"sub-{int(subject):03}_meta-data.json")
# with open(metadata_path, 'r') as f:
#     metadata = json.load(f)

# behav_files = get_fileslist(os.path.join(SECOND_ANALYSIS__DATA_DIR, f"sub-{subject:03}", "raw", "beh"), "stratinf_beh.tsv")
# beh_df = pd.read_csv(behav_files[0], sep="\t")
# events_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "raw", "ieeg", f"sub-{int(subject):03}_events.tsv")
# events = pd.read_csv(events_path, sep = "\t")
# eeg_alignement = events["align_eeg"].values.astype(str)
# eeg_stim_count = sum(s.count(c) for s in eeg_alignement for c in '123')
# print(metadata["n_epochs"], eeg_stim_count)




# subject = SUBJECTS[3]
# for subject in SUBJECTS[7:]:
subject = 2
    print(f"Processing subject {subject}")
    epochs_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "epochs", f"sub-{int(subject):03}_ieeg-epochs.fif")
    epochs = mne.read_epochs(epochs_path, preload = True, verbose='error')
    ch_names = epochs.info["ch_names"]
    sfreq = epochs.info["sfreq"]

    n_epochs = len(epochs)
    n_channels = len(ch_names)

    power_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-power.npy")
    phase_path = os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-phase.npy")

    power_complete = np.memmap(power_path, dtype = 'float16',mode = 'w+', shape = (n_epochs, n_channels, n_freqs, n_times_decimed))
    phase_complete = np.memmap(phase_path, dtype = 'float16',mode = 'w+', shape = (n_epochs, n_channels, n_freqs, n_times_decimed))
    data = epochs.get_data(copy=True)
    del epochs
    gc.collect()

    for start_idx in tqdm(range(0, n_epochs, 20)) :
        end_idx = min(start_idx + 20, n_epochs)
        batch_data = data[start_idx:end_idx, :, :]
        tfr = mne.time_frequency.tfr_array_multitaper(batch_data, sfreq, freqs=freqlist, n_cycles=cycles, time_bandwidth=3.0, use_fft=True, decim=decimation, n_jobs=-1, output='complex', verbose=False)
        
        power = np.abs(tfr) ** 2
        power_avg = np.mean(power, axis=2) 
        power_db = 10 * np.log10(power_avg)
        power_complete[start_idx:end_idx] += power_db

        phase = np.angle(tfr)
        phase_avg_tapers = np.angle(np.mean(np.exp(1j * phase), axis=2))
        phase_complete[start_idx:end_idx] += phase_avg_tapers
        
        power_complete.flush()
        phase_complete.flush()

        del power, power_avg, power_db, tfr, batch_data, phase, phase_avg_tapers
        gc.collect()


    del data
    gc.collect()

    meta_data = {
        'sfreq': sfreq,
        'n_epochs': n_epochs,
        'n_channels': n_channels,
        'n_times': n_times,
        'ch_names': ch_names,
        'n_freqs' : n_freqs,
        'freqlist': freqlist.tolist(),
        'cycles': cycles.tolist(),
        'times_decimed': times_decimed.tolist(),
        'n_times_decimed': n_times_decimed,
        'time_bandwidth': 3.0,
        'decim': decimation,
        'output': 'complex'
    }

    metadata_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}", "preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-metadata.json"))
    with open(metadata_path, 'w', encoding='utf-8') as f: 
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    baseline_path = os.path.join(os.path.join(DATA_DIR, f"sub-{int(subject):03}","preprocessed", "timefreq", f"sub-{int(subject):03}_tfr-baseline.npy"))
    baseline_power = np.mean(power_complete, axis=-1, keepdims=True)
    np.save(baseline_path, baseline_power)
    del power_complete, phase_complete, baseline_power
    gc.collect()

