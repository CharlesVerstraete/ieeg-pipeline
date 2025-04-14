#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Custom import for subject 25, edf file from deltamed system (Bordeaux)
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *

new_action_trigger = {7: 71, 9: 73, 15: 79, 186: 250, 187: 251, 188: 252, 37: 101, 136: 200, 38: 102, 138: 202, 146: 210, 148: 212, 36: 100, 189: 253, 137: 201, 46: 110, 147: 211, 48: 112, 47: 111, 176: 240, 23: 151}

subject = 25
create_subject_folder(subject)
raw_data_folder = os.path.join(ORIGINAL_DATA_DIR, str(subject), 'EEG')
ieeg_files = get_fileslist(raw_data_folder, '.edf')
for (i, ieeg_file) in enumerate(ieeg_files):
    idx = i + 1
    raw = mne.io.read_raw_edf(ieeg_file, preload=True)
    events, events_id = mne.events_from_annotations(raw)
    events_id_reverse = {v: k for k, v in events_id.items()}

    cleaned_events = [events_id_reverse[x].split(" ")[-1] for x in events[:, 2]]

    signal_events = pd.DataFrame({
        "sample": events[:, 0],
        "value": cleaned_events
    })
    signal_events["is_trigger"] = [x.isdigit() for x in signal_events["value"]]

    filtered_signal_events = signal_events[signal_events["is_trigger"]].copy().reset_index(drop=True)
    filtered_signal_events["value"] = filtered_signal_events["value"].astype(int)

    filtered_signal_events["new_value"] = filtered_signal_events["value"].replace(new_action_trigger)

    new_events = np.stack([filtered_signal_events["sample"].values, np.zeros(len(filtered_signal_events)), filtered_signal_events["new_value"].values], axis=1)
    raw = add_stim_events(raw, new_events)

    if raw.info['sfreq'] < SAMPLING_RATE:
        raw, new_events = raw.resample(SAMPLING_RATE, events = new_events, n_jobs=-1)
    save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'ieeg', f'sub-{subject:03d}_run-{idx:02d}-raw.fif')
    raw.save(save_path, overwrite=True)







############################################################################################################################################################################
############################################################################################################################################################################
# Finding correspondance for errors in marker value (custom parrallel to dvi cable)
############################################################################################################################################################################
############################################################################################################################################################################


trigger_table = pd.read_csv("/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/25/EEG/25_2025_01_17_10_53_trigTable.csv")
trigger_table["sample"] = (trigger_table["time"] * raw.info['sfreq']).astype(int)

centered_signal_events = filtered_signal_events[15:].copy().reset_index(drop=True)
centered_log_events = trigger_table[12:].copy().reset_index(drop=True)

centered_signal_events["centered_sample"] = centered_signal_events["sample"] - centered_signal_events["sample"][0]
centered_log_events["centered_sample"] = centered_log_events["sample"] - centered_log_events["sample"][0]

centered_log_events["matched_signal"] = 0
centered_log_events["match_distance"] = 0
centered_log_events["match_value"] = 0
centered_log_events["match_idx"] = 0



for i in range(len(centered_log_events)):
    log_value = centered_log_events["centered_sample"].values[i]
    distances = np.abs(centered_signal_events["centered_sample"] - log_value)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    centered_log_events.loc[i, "matched_signal"] = centered_signal_events["centered_sample"][best_idx]
    centered_log_events.loc[i, "match_distance"] = best_distance
    centered_log_events.loc[i, "match_value"] = centered_signal_events["value"].values[best_idx]
    centered_log_events.loc[i, "match_idx"] = int(best_idx)


centered_log_events = centered_log_events[:2512].copy().reset_index(drop=True)
centered_log_events["is_same_value"] = centered_log_events["value"].values == centered_log_events["match_value"].values
bad_triggers = centered_log_events[~centered_log_events["is_same_value"]].copy().reset_index(drop=True)

bad_matching = {x : [] for x in bad_triggers["match_value"].values}
for idx, row in bad_triggers.iterrows():
    bad_matching[row["match_value"]].append(row["value"])

bad_matching.pop(23)
new_action_trigger = {k : np.unique(v)[0] for k, v in bad_matching.items()}
