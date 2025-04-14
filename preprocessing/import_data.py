#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Import data from the original raw dataset and save it in BIDS-like format
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *

# # if __name__ == "__main__":

for subject in SUBJECTS :
    create_subject_folder(subject)
    collect_raw_data(subject)



# custom for subject 8 (analog triggers)

subject = 8
raw_data_folder = os.path.join(ORIGINAL_DATA_DIR, str(subject), 'EEG')
raw_files = os.listdir(raw_data_folder)
ieeg_files = get_fileslist(raw_data_folder, '.trc')
ieeg_file = ieeg_files[0]

signal, channel_signal, electrodes_types, events, sfreq = import_trc_file(ieeg_file, subject)

trigger_table = pd.read_csv("/Users/charles.verstraete/Documents/w3_iEEG/subject_collection/8/EEG/8_2024_03_19_16_14_trigTable.csv")

stim_idx = np.where(channel_signal == 'sti2+')[0][0]
stim_channel = signal[:, stim_idx]
plt.scatter(np.arange(len(stim_channel)), stim_channel)
plt.show()

events_finder = np.zeros(len(stim_channel))
events_finder[stim_channel > 0.001] = 1
consecutive = find_consecutive(events_finder)

sample_events = np.array([x[np.argmax(stim_channel[x])] for x in consecutive if ((len(x) < 13) & (len(x) > 7))])
events_found = np.zeros(len(stim_channel))
events_found[sample_events] = 1

plt.scatter(np.arange(len(events_found)), events_found, s=1)
plt.plot(stim_channel*1000, color = "red", alpha = 0.5)
plt.show()

time_events = sample_events/sfreq
centered_events = time_events - time_events[0]

log_time_events = trigger_table["time"].values
log_centered_events = log_time_events - log_time_events[0]
log_sample_events = (log_time_events * sfreq).astype(int)

signal_center_sample = sample_events[30:] - sample_events[30]
log_center_sample = log_sample_events[21:] - log_sample_events[21]

plt.scatter(np.arange(len(log_sample_events)), log_sample_events - log_sample_events[0], color = "blue", alpha = 0.5)
plt.scatter(np.arange(len(sample_events)), sample_events, color = "red", alpha = 0.5)
plt.show()

sub_test_df = trigger_table.loc[21:].copy().reset_index()

sub_test_df["centered_sample"] = log_center_sample
sub_test_df["matched_signal"] = np.nan
sub_test_df["match_distance"] = np.nan
sub_test_df["match_idx"] = np.nan

for i in range(len(sub_test_df)):
    log_value = sub_test_df["centered_sample"].values[i]
    distances = np.abs(signal_center_sample - log_value)
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    sub_test_df.loc[i, "matched_signal"] = signal_center_sample[best_idx]
    sub_test_df.loc[i, "match_distance"] = best_distance
    sub_test_df.loc[i, "match_idx"] = int(best_idx)


test_idx = sub_test_df[sub_test_df["good_match"]]["match_idx"].values.astype(int)
trigger_values = sub_test_df[sub_test_df["good_match"]]["value"].values.astype(int)

idx_events = sample_events[test_idx+30]
events = np.stack((idx_events, np.zeros(len(idx_events)), trigger_values), axis = 1)

raw, events = create_mne(signal, channel_signal, electrodes_types, sfreq, events)

raw.plot(events=events)
raw.info


















first_sample_signal = np.where(stim_channel > 1000)[0][0]




consecutive = find_consecutive(events_finder)
consecutive_length = [len(x) for x in consecutive]
sample_events = [int(np.mean(x)) for x in consecutive]

sample_events = np.array([x[0] for x in consecutive])
time_events = sample_events/sfreq
centered_events = time_events - time_events[0]

log_time_events = trigger_table["time"].values
log_centered_events = log_time_events - log_time_events[0]
log_sample_events = log_centered_events * sfreq

sample_events

plt.scatter(centered_events, np.zeros(len(centered_events)), label = "MNE events", s=1)
plt.scatter(log_centered_events, np.ones(len(log_centered_events)), label = "Log events", s= 1)
plt.legend()
plt.show()

centered_events[-3825:]
np.where(centered_events<2440)
log_centered_events[145]
centered_events[146]

recentered_true = centered_events[118:1546] - centered_events[118]
recentered_log = log_centered_events[18:1446] - log_centered_events[18]


plt.scatter(recentered_true, np.zeros(len(recentered_true)), label = "MNE events", s=1)
plt.scatter(recentered_log, np.ones(len(recentered_log)), label = "Log events", s= 1)
plt.legend()
plt.show()

plt.plot(recentered_true - recentered_log)
plt.show()
recentered_true[1231]
recentered_log[1230]


trigger_table.loc[1245:1250]

events_check_df = trigger_table.copy()

events_check_df["centered_time"] = events_check_df["time"] - events_check_df["time"].values[0]
events_check_df["centered_sample"] = (events_check_df["centered_time"] * sfreq).astype(int)

sample_events


from scipy.optimize import minimize


def objective_function(offset):
    aligned = s_segment + offset
    return np.sum((aligned - l_segment)**2)

# Initial guess: scale=1, offset=0
initial_guess = [1.0, 0.0]
result = minimize(objective_function, initial_guess)


log_times = log_centered_events
signal_times = centered_events
max_shift = 1000

signal_times = np.array(signal_times)
log_times = np.array(log_times)

# Try different alignments to find best starting point match
best_error = float('inf')
best_shift = 0
best_offset = 0.0

# Try different shifts to find best alignment
for shift in range(-max_shift, max_shift + 1):
    if shift < 0:
        # Signal starts after log
        s_idx = 0
        l_idx = -shift
    else:
        # Signal starts before log
        s_idx = shift
        l_idx = 0
        
    # Calculate overlap length
    s_end = min(len(signal_times) - s_idx, len(log_times) - l_idx)
    if s_end < 10:  # Require minimum overlap
        continue
        
    # Get overlapping segments
    s_segment = signal_times[s_idx:s_idx + s_end]
    l_segment = log_times[l_idx:l_idx + s_end]
    
    # Re-center segments to improve numerical stability
    s_segment = s_segment - s_segment[0]
    l_segment = l_segment - l_segment[0]

    params = {"offset": 0.0, "s_segment": s_segment, "l_segment": l_segment}
    result = minimize(objective_function, 0.0)
    offset = result.x
    error = result.fun

    if error < best_error:
        best_error = error
        best_shift = shift
        best_offset = offset



aligned_events = log_times + offset
aligned_events

plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(log_centered_events)), log_centered_events, label="Log events", s=10)
plt.scatter(np.arange(len(centered_events))+ best_shift, centered_events, label="Aligned signal events", s=10)
plt.legend()
plt.title("Event Alignment Results")
plt.xlabel("Event index")
plt.ylabel("Time (s)")
plt.show()




align_events(signal_times, log_times)



centered_sample_events = sample_events - sample_events[0]

centered_sample_events[119] - centered_sample_events[118]
events_check_df["centered_sample"][19] - events_check_df["centered_sample"][18]

sub_test_df = events_check_df.loc[21:].copy().reset_index()
centered_sample_events_test = centered_sample_events[121:] - centered_sample_events[121]
centered_sample_events_test

sub_test_df["centered_sample"] = sub_test_df["centered_sample"] - sub_test_df["centered_sample"].values[0]
sub_test_df["test"] = 0
for i in range(len(sub_test_df)):
    log_value = sub_test_df["centered_sample"].values[i]
    signal_value = centered_sample_events_test[i]
    if np.abs(log_value-signal_value) < 2:
        sub_test_df["test"].values[i] = signal_value
    else:
        j = 0
        while j < len(centered_sample_events_test) and np.abs(log_value-signal_value) > 3:
            signal_value = centered_sample_events_test[j]
            j += 1
        if j < len(centered_sample_events_test):
            sub_test_df["test"].values[i] = centered_sample_events_test[j]




sub_test_df["matched_signal"] = np.nan
sub_test_df["match_distance"] = np.nan
sub_test_df["match_idx"] = np.nan


# For each log event, find best matching signal event
for i in range(len(sub_test_df)):
    log_value = sub_test_df["centered_sample"].values[i]
    
    # Calculate distances to all signal events
    distances = np.abs(centered_sample_events_test - log_value)
    
    # Find the index of the closest signal event
    best_idx = np.argmin(distances)
    best_distance = distances[best_idx]
    
    # Store the matched value and the distance
    sub_test_df.loc[i, "matched_signal"] = centered_sample_events_test[best_idx]
    sub_test_df.loc[i, "match_distance"] = best_distance
    sub_test_df.loc[i, "match_idx"] = int(best_idx)

test_idx = sub_test_df["match_idx"].values.astype(int)






centered_sample_events_test.shape

# Analyze the quality of matches
print(f"Mean distance: {sub_test_df['match_distance'].mean():.2f} samples")
print(f"Max distance: {sub_test_df['match_distance'].max():.2f} samples")
print(f"Matches within 2 samples: {(sub_test_df['match_distance'] <= 10).sum()} of {len(sub_test_df)}")

# Flag potentially problematic matches (distance > threshold)
threshold = 10
sub_test_df["good_match"] = sub_test_df["match_distance"] <= threshold
plt.plot(sub_test_df["match_distance"])
plt.show()
# Plot the matches
plt.figure(figsize=(12, 6))
plt.scatter(sub_test_df.index, sub_test_df["centered_sample"], label="Log events")
plt.scatter(sub_test_df.index, sub_test_df["matched_signal"], label="Matched signal events")
plt.legend()
plt.title("Event Matching Results")
plt.xlabel("Event index")
plt.ylabel("Sample number")
plt.show()
sub_test_df
sub_test_df[sub_test_df["test"] == 0]
centered_sample_events_test
sub_test_df["sample_signal"] = centered_sample_events[121:121+len(sub_test_df)]
sub_test_df["centered_sample_signal"] = sub_test_df["sample_signal"] - sub_test_df["sample_signal"].values[0]
sub_test_df["diff"] = sub_test_df["centered_sample_signal"] - sub_test_df["centered_sample"]

sub_test_df[sub_test_df["diff"] > 1]


recentered_true[:50]
recentered_log[:50]
np.where((recentered_true - recentered_log)>0.1)


np.where(segment.analogsignals[0].array_annotations["channel_names"] == "sti2+")
segment.analogsignals[0].array_annotations["channel_names"]
segment.analogsignals[0][:, 84]
plt.plot(segment.analogsignals[0][:, 84])
plt.show()

raw.add_events(np.array(events), stim_channel='STIM')



save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'ieeg', f'sub-{subject:03d}_run-{idx:02d}-raw.fif')
raw.save(save_path, overwrite=True)
print(" "*10 + f"Saving {ieeg_file.split('/')[-1]} as sub-{subject:03d}_run-{idx:02d}-raw.fif")


