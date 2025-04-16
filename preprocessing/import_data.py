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
from preprocessing.utils.align_helper import *
# # if __name__ == "__main__":

# for subject in SUBJECTS :
#     create_subject_folder(subject)
#     collect_raw_data(subject)


for subject in [2, 4, 5, 8, 9, 14, 16, 19, 20, 25] : 
    print(f"{subject}")
    if subject == 25 :
        signal_events_df = get_complete_events(subject)
        clean_signal_events = clean_eventdf(signal_events_df[3:])
    else :
        signal_events_df = get_complete_events(subject)
        clean_signal_events = clean_eventdf(signal_events_df)

    events_file = get_fileslist(os.path.join(ORIGINAL_DATA_DIR, str(subject), 'EEG'), '.csv')[0]
    log_events_df = pd.read_csv(events_file)
    clean_log_events = clean_eventdf(log_events_df)

    behav_files = get_fileslist(os.path.join(SECOND_ANALYSIS__DATA_DIR, f"sub-{subject:03}", "raw", "beh"), "stratinf_beh.tsv")
    beh_df = pd.read_csv(behav_files[0], sep="\t")
    beh_events, trial_count = get_behav_events(beh_df)
    beh_events_df = pd.DataFrame({"value" : beh_events, "trial_count" : trial_count})
    beh_events_df["cumul"] = np.arange(len(beh_events_df))
    beh_events_df["format_events"] = list(format_events(beh_events))

    format_log_events = format_events(clean_log_events["value"].values)
    format_eeg_events = format_events(clean_signal_events["value"].values)
    format_behav_events = format_events(beh_events)

    aligned_behav_log, aligned_log_behav = needleman_wunsch(format_behav_events, format_log_events)
    aligned_log_eeg, aligned_eeg_log = needleman_wunsch(format_log_events, format_eeg_events)
    aligned_behav_eeg, aligned_eeg_behav = needleman_wunsch(format_behav_events, format_eeg_events)

    events_df = make_eventsdf(beh_events_df, aligned_log_behav, aligned_eeg_behav)
    events_df = check_alignement(events_df, clean_signal_events, clean_log_events, aligned_log_eeg, aligned_eeg_behav, aligned_behav_eeg)

    missing_trials_df = events_df[events_df["align_eeg"] == "-"].copy()
    print(missing_trials_df)

    save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'ieeg', f'sub-{subject:03d}_events.tsv')
    events_df.to_csv(save_path, sep="\t", index=False)

