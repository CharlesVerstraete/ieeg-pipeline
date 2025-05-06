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
    # collect_raw_data(subject)



for subject in [4, 5, 9, 12, 14, 16, 19, 20, 23, 28] : 
    clean_signal_events, clean_log_events, beh_events_df = get_triplet_eventsdf(subject)
    aligned_behav_eeg, aligned_log_behav, aligned_eeg_behav = create_alignseq(clean_log_events, clean_signal_events, beh_events_df)
    events_df = make_eventsdf(beh_events_df, aligned_log_behav, aligned_eeg_behav)
    events_df = check_alignement(events_df, clean_signal_events, clean_log_events)
    missing_trials_df = events_df[events_df["align_eeg"] == "-"].copy()
    print(missing_trials_df)
    save_path = os.path.join(DATA_DIR, f'sub-{subject:03}', 'raw', 'ieeg', f'sub-{subject:03d}_events.tsv')
    events_df.to_csv(save_path, sep="\t", index=False)




####

# subject = 3
# clean_signal_events, clean_log_events, beh_events_df = get_triplet_eventsdf(subject)
# aligned_behav_eeg, aligned_log_behav, aligned_eeg_behav = create_alignseq(clean_log_events, clean_signal_events, beh_events_df)


# format_log_events = format_events(clean_log_events["value"].values)
# format_behav_events = format_events(beh_events_df["value"].values)
# format_eeg_events = format_events(clean_signal_events["value"].values)

# aligned_behav_log, aligned_log_behav = needleman_wunsch(format_behav_events, format_log_events)
# aligned_behav_eeg, aligned_eeg_behav = needleman_wunsch(format_behav_events, format_eeg_events)

# events_df = pd.DataFrame({
#     "behav_seq" : list(aligned_behav_eeg),
#     "align_eeg" :  list(aligned_eeg_behav),
#     "align_log" :  np.nan,
#     "trigger_value" : np.nan,
#     "trial_count" : np.nan,
#     "cumul_seq" : np.nan,
#     "value_log" : np.nan,
#     "value_eeg" : np.nan,
#     "time_signal" : np.nan,
#     "time_log" : np.nan,
#     "sample" : np.nan,
#     "run" : np.nan
# })

# events_df.loc[(events_df["behav_seq"] != "-"), "align_log"] = list(aligned_log_behav)
# events_df.loc[(events_df["behav_seq"] != "-"), "trigger_value"] = beh_events_df["value"].values
# events_df.loc[(events_df["behav_seq"] != "-"), "trial_count"] = beh_events_df["trial_count"].values
# events_df.loc[(events_df["behav_seq"] != "-"), "cumul_seq"] = beh_events_df["cumul"].values
# events_df.loc[((events_df["align_log"] != "-") & (events_df["behav_seq"] != "-")), "value_log"] = clean_log_events["value"].values
# events_df.loc[(events_df["align_eeg"] != "-"), "value_eeg"] = clean_signal_events["value"].values
# events_df.loc[(events_df["align_eeg"] != "-"), "time_signal"] = clean_signal_events["time"].values
# events_df.loc[((events_df["align_log"] != "-") & (events_df["behav_seq"] != "-")), "time_log"] = clean_log_events["time"].values
# events_df.loc[(events_df["align_eeg"] != "-"), "sample"] = clean_signal_events["sample"].values
# events_df.loc[(events_df["align_eeg"] != "-"), "run"] = clean_signal_events["run"].values

