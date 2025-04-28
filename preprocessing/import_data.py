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


for subject in SUBJECTS : 
    clean_signal_events, clean_log_events, beh_events_df = get_triplet_eventsdf(subject)
    aligned_behav_eeg, aligned_log_behav, aligned_eeg_behav = create_alignseq(clean_log_events, clean_signal_events, beh_events_df)
    events_df = make_eventsdf(beh_events_df, aligned_log_behav, aligned_eeg_behav)
    events_df = check_alignement(events_df, clean_signal_events, clean_log_events)
    missing_trials_df = events_df[events_df["align_eeg"] == "-"].copy()
    print(missing_trials_df)

    save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'ieeg', f'sub-{subject:03d}_events.tsv')
    events_df.to_csv(save_path, sep="\t", index=False)




