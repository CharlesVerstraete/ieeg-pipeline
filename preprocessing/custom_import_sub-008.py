
#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Custom import for subject 8, analogue triggers from micromed .trc 
Stimulus channel : sti2+
Aligning events with trigger table
7 sample drift during the 3500 seconds recording
First sample index in task 30 in .trc, 21 in trigger table
Event 1228 in realign trigger table is a duplicated sample (6 sample answer from subject not passed in system)
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *


# Import data 
subject = 8
create_subject_folder(subject)
raw_data_folder = os.path.join(ORIGINAL_DATA_DIR, str(subject), 'EEG')
raw_files = os.listdir(raw_data_folder)
ieeg_files = get_fileslist(raw_data_folder, '.trc')
ieeg_file = ieeg_files[0]
idx = 1

signal, channel_signal, electrodes_types, events, sfreq = import_trc_file(ieeg_file, subject)

trigger_table = pd.read_csv("/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/8/EEG/8_2024_03_19_16_14_trigTable.csv")

# Get stim index and look at how the triggers are distributed
stim_idx = np.where(channel_signal == 'sti2+')[0][0]
stim_channel = signal[:, stim_idx]
plt.scatter(np.arange(len(stim_channel)), stim_channel)
plt.show()

# Get sample where there is a non null signal
events_finder = np.zeros(len(stim_channel))
events_finder[stim_channel > 0.001] = 1
consecutive = find_consecutive(events_finder)

# Get the length of each consecutive event and get the sample index where the signal is the highest
sample_events = np.array([x[np.argmax(stim_channel[x])] for x in consecutive if ((len(x) < 13) & (len(x) > 7))])
events_found = np.zeros(len(stim_channel))
events_found[sample_events] = 1

plt.scatter(np.arange(len(events_found)), events_found, s=1)
plt.plot(stim_channel*1000, color = "red", alpha = 0.5)
plt.show()

# Center the events from signal and log
time_events = sample_events/sfreq
centered_events = time_events - time_events[0]

trigger_table["sample"] = (trigger_table["time"] * sfreq).astype(int)

log_time_events = trigger_table["time"].values
log_sample_events = trigger_table["sample"].values

signal_center_sample = sample_events[30:] - sample_events[30]
log_center_sample = log_sample_events[21:] - log_sample_events[21]

plt.scatter(np.arange(len(signal_center_sample)), signal_center_sample, color = "blue", alpha = 0.5, label = "Signal events")
plt.scatter(np.arange(len(log_center_sample)), log_center_sample, color = "red", alpha = 0.5, label = "Log events")
plt.legend()
plt.show()


# Align the events from the signal and the log
align_trigger_table = trigger_table.loc[21:].copy().reset_index(drop=True)

align_trigger_table["centered_sample"] = align_trigger_table["sample"] - align_trigger_table["sample"].values[0]
align_trigger_table["matched_signal"] = np.nan
align_trigger_table["match_distance"] = np.nan
align_trigger_table["match_idx"] = np.nan

for i in range(len(align_trigger_table)):
    log_value = align_trigger_table["centered_sample"].values[i]
    distances = np.abs(signal_center_sample - log_value)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    align_trigger_table.loc[i, "matched_signal"] = signal_center_sample[best_idx]
    align_trigger_table.loc[i, "match_distance"] = best_distance
    align_trigger_table.loc[i, "match_idx"] = int(best_idx)


threshold = 8
align_trigger_table["good_match"] = align_trigger_table["match_distance"] <= threshold
align_trigger_table.loc[1228, "good_match"] = False

plt.plot(align_trigger_table["good_match"])
plt.show()

test_idx = align_trigger_table[align_trigger_table["good_match"]]["match_idx"].values.astype(int)
trigger_values = align_trigger_table[align_trigger_table["good_match"]]["value"].values.astype(int)

idx_events = sample_events[test_idx+30]
events = np.stack((idx_events, np.zeros(len(idx_events)), trigger_values), axis = 1)

# Add the events to the raw object and save it
raw, events = create_mne(signal, channel_signal, electrodes_types, sfreq, events)

save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}',  'raw', 'ieeg', f'sub-{subject:03d}_run-{idx:02d}-raw.fif')
raw.save(save_path, overwrite=True)


