#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Helper functions for aligning data and events dataframe making
"""
# Import libraries

import numpy as np
import os
import numpy as np
import pandas as pd 
import mne

from preprocessing.config import *
from preprocessing.utils.data_helper import get_fileslist
# Functions definitions

def behav_transform(stim, choice, correct, fb, is_stimstable, trap, miss, trial_count):
    """
    Transform the behavioral trial into a sequence of corresponding integer between 0 and 9
    """
    s = stim * 10 
    s += (2 if is_stimstable==1 else 1 if is_stimstable==0 else 0)
    
    if miss == 1:
        seq = [int(s), 253]
        seq_trial = [int(trial_count)] * 2
    else : 
        c = 100*(correct + 1) + 10*trap + (choice-1)
        o = 153 - 2*fb + trap
        seq = [int(s), int(c), int(o)]
        seq_trial = [int(trial_count)] * 3
    return seq, seq_trial

def remove_start(df):
    """
    Remove the start of the experiment.
    """
    # start_idxs = np.where(df["value"] == 252)[0]
    # if len(start_idxs) == 0:
    #     start_idx = 0
    # else:
    #     if start_idxs[1] - start_idxs[0] < 4:
    #         start_idx = start_idxs[1]
    #     else :
    #         start_idx = start_idxs[0]
    bad_idx = np.where(np.array(list(format_events(df["value"].values))) == '0')[0]
    if len(bad_idx) > 0 :
        if bad_idx[0] != 0 :
            tmpd_df = df.iloc[bad_idx[0]:].reset_index(drop=True)
        else :
            tmpd_df = df.copy()
    else :
        tmpd_df = df.copy()
    good_idx = np.where(np.array(list(format_events(tmpd_df["value"].values))) != '0')[0]
    result_df = tmpd_df.loc[good_idx].reset_index(drop=True)
    return result_df

def remove_init(df):
    """
    Remove the end of the experiment.
    """
    return df[~df["value"].isin([71, 73, 79])].reset_index(drop=True)

def remove_breaks(df):
    """
    Remove the breaks in the experiment.
    """
    break_idx = np.where(df["value"] == 240)[0]
    idx_torm = []
    if len(break_idx) != 0:
        for idx in break_idx:
            space_to_end = len(df) - idx
            new_start = np.where(df["value"][idx:] == 252)[0]
            new_start = new_start[1] + 1 if len(new_start) > 1 else space_to_end
            space_remove = min(space_to_end, new_start)
            idx_torm.extend(range(idx, idx + space_remove))
    return df.drop(idx_torm).reset_index(drop=True)

def remove_notrials(df):
    """
    Remove the trials in the experiment.
    """
    return df[~df["value"].isin([250, 251, 252, 255])].reset_index(drop=True)

def clean_eventdf(df):
    """
    Clean the event dataframe.
    """
    # df = remove_init(df)
    df = remove_breaks(df)
    df = remove_start(df)
    # df = remove_notrials(df)
    return df


def get_behav_events(df):
    """
    Get the events from a behavioral file.
    """
    trial_count = []
    events = []
    for row in df[['stim', 'choice', 'correct', 'fb', 'is_stimstable', 'trap', 'miss', 'trial_count']].values : 
        tmp_events, tmp_trial_count = behav_transform(*row)
        events.append(tmp_events)
        trial_count.append(tmp_trial_count)

    return np.hstack(events), np.hstack(trial_count)

def needleman_wunsch(seq1, seq2, match_score=5, mismatch_score=-3, gap_open_seq1=-100, gap_extend_seq1=-100, gap_open_seq2=-20, gap_extend_seq2=-1, band_width=2000):
    """
    Needleman-Wunsch algorithm for global sequence alignment with banded constraints.
    """
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1), dtype=int)
    
    score_matrix[:, 0] = gap_open_seq1 + np.arange(m + 1) * gap_extend_seq1
    score_matrix[0, :] = gap_open_seq2 + np.arange(n + 1) * gap_extend_seq2

    for i in range(1, m + 1):
        start_j, end_j = max(1, i - band_width), min(n + 1, i + band_width + 1)
        for j in range(start_j, end_j):
            match = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            score_matrix[i, j] = max(
                score_matrix[i - 1, j - 1] + match,
                score_matrix[i - 1, j] + (gap_open_seq1 if j == start_j else gap_extend_seq1),
                score_matrix[i, j - 1] + (gap_open_seq2 if i == max(1, j - band_width) else gap_extend_seq2)
            )

    align1, align2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i - 1, j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score):
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i - 1, j] + (gap_open_seq1 if j == max(1, i - band_width) else gap_extend_seq1):
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1
    
    return ''.join(reversed(align1)), ''.join(reversed(align2))

# def needleman_wunsch(seq1, seq2, match_score=5, mismatch_score=-3, gap_open_seq1=-1000, gap_extend_seq1=-1000, gap_open_seq2=-10, gap_extend_seq2=-1, band_width=2000):
#     """
#     Needleman-Wunsch algorithm for global sequence alignment with banded constraints.
#     """
#     m, n = len(seq1), len(seq2)
#     score_matrix = np.zeros((m + 1, n + 1), dtype=int)
#     score_matrix[1:, 0] = gap_open_seq1 + np.arange(m) * gap_extend_seq1
#     score_matrix[0, 1:] = gap_open_seq2 + np.arange(n) * gap_extend_seq2
#     for i in range(1, m + 1):
#         start_j = max(1, i - band_width)
#         end_j = min(n + 1, i + band_width + 1)
#         for j in range(start_j, end_j):
#             match = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
#             diag_score = score_matrix[i-1, j-1] + match

#             is_first_gap_seq1 = j == start_j or score_matrix[i-1, j] != score_matrix[i-2, j] + gap_extend_seq1
#             gap_penalty_seq1 = gap_open_seq1 if is_first_gap_seq1 else gap_extend_seq1
#             up_score = score_matrix[i-1, j] + gap_penalty_seq1
            
#             is_first_gap_seq2 = i == start_j or score_matrix[i, j-1] != score_matrix[i, j-2] + gap_extend_seq2
#             gap_penalty_seq2 = gap_open_seq2 if is_first_gap_seq2 else gap_extend_seq2
#             left_score = score_matrix[i, j-1] + gap_penalty_seq2
            
#             score_matrix[i, j] = max(diag_score, up_score, left_score)
    
#     align1, align2 = [], []
#     i, j = m, n
    
#     while i > 0 or j > 0:
#         if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
#             align1.append(seq1[i-1])
#             align2.append(seq2[j-1])
#             i -= 1
#             j -= 1
#         elif i > 0 and (j == 0 or score_matrix[i, j] == score_matrix[i-1, j] + (gap_open_seq1 if i == 1 or score_matrix[i-1, j] != score_matrix[i-2, j] + gap_extend_seq1 else gap_extend_seq1)):
#             align1.append(seq1[i-1])
#             align2.append('-')
#             i -= 1
#         else:
#             align1.append('-')
#             align2.append(seq2[j-1])
#             j -= 1
    
#     return ''.join(reversed(align1)), ''.join(reversed(align2))


def get_events_fromsignal(raw, run) :
    """
    Get the events from the raw signal
    """
    events = mne.find_events(raw, verbose=False)   
    df = pd.DataFrame(events[:, [0, 2]], columns = ["sample", "value"])
    df["run"] = run
    df["time"] = df["sample"] / raw.info['sfreq']
    return df

def get_complete_events(subject) :
    """
    Get complete df from all the runs of the subject 
    """
    run = 1
    raw_path = os.path.join(DATA_DIR, f'sub-{subject:03d}',  'raw', 'ieeg', f'sub-{subject:03d}_run-{run:02d}-raw.fif')
    signal_events = pd.DataFrame()
    while os.path.exists(raw_path) :
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
        tmp = get_events_fromsignal(raw, run)
        signal_events = pd.concat([signal_events, tmp], ignore_index=True)
        run += 1
        raw_path = os.path.join(DATA_DIR, f'sub-{subject:03d}',  'raw', 'ieeg', f'sub-{subject:03d}_run-{run:02d}-raw.fif')
    signal_events = signal_events.reset_index(drop=True)
    del raw, tmp
    return signal_events

def format_events(events):
    """
    Convert the events to a string.
    """
    formated = events.copy()
    event_map = {
        (10, 11, 12): 1, (20, 21, 22): 2, (30, 31, 32): 3,
        (100, 110, 200, 210): 4, (101, 111, 201, 211): 5, (102, 112, 202, 212): 6,
        (151, 152): 7, (153, 154): 8, (253,): 9
    }
    for values, new_value in event_map.items():
        formated[np.isin(events, values)] = new_value
    formated[formated > 9] = 0
    return ''.join(map(str, formated))


# def format_events(events):
#     """
#     Convert the events to a string.
#     """
#     formated = np.zeros(len(events), dtype=str)
#     event_map = {
#         (10): 'A', (11): 'B', (12): 'C',
#         (20): 'D', (21): 'E', (22): 'F',
#         (30): 'G', (31): 'H', (32): 'I',
#         (100): 'J', (110): 'K', (200): 'L', (210): 'M',
#         (101): 'N', (111): 'O', (201): 'P', (211): 'Q',
#         (102): 'R', (112): 'S', (202): 'T', (212): 'U',
#         (151): 'V', (152): 'W',
#         (153): 'X', (154): 'Y',
#         (253): 'Z'
#     }

#     for values, new_value in event_map.items():
#         idx = np.where(events == values)[0]
#         if len(idx) > 0:
#             formated[idx] = new_value
#     check = np.where(formated == '')[0]
#     if len(check) > 0:
#         formated[check] = "0"
#     return ''.join(map(str, formated))


def get_triplet_eventsdf(subject): 
    """
    
    """
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

    return clean_signal_events, clean_log_events, beh_events_df

def create_alignseq(log_events, signal_events, beh_events): 
    """
    
    """
    format_log_events = format_events(log_events["value"].values)
    format_eeg_events = format_events(signal_events["value"].values)
    format_behav_events = format_events(beh_events["value"].values)

    aligned_behav_log, _ = needleman_wunsch(format_behav_events, format_log_events)
    aligned_behav_eeg, aligned_eeg_behav = needleman_wunsch(format_behav_events, format_eeg_events)

    return aligned_behav_log, aligned_behav_eeg, aligned_eeg_behav


def check_alignement(events_df, signal_df, log_df) :
    """
    Check the alignment of the events dataframe with the signal and log dataframe
    """
    itr = 0
    for idx, row in events_df.iterrows() :
        if "-" not in row.values : 
            events_df.loc[idx, "time_signal"] = signal_df["time"].values[itr]
            events_df.loc[idx, "time_log"] = log_df["time"].values[idx]
            events_df.loc[idx, "sample"] = signal_df["sample"].values[itr]
            events_df.loc[idx, "run"] = signal_df["run"].values[itr]
            events_df.loc[idx, "value_log"] = log_df["value"].values[idx]
            events_df.loc[idx, "value_eeg"] = signal_df["value"].values[itr]
            itr += 1
    return events_df

def make_eventsdf(beh_events_df, aligned_log_behav, aligned_eeg_behav) : 
    """
    Make the events dataframe from the aligned log and eeg events
    """
    events_df = pd.DataFrame({
        "behav_seq" : beh_events_df["format_events"],
        "align_log" : list(aligned_log_behav),
        "align_eeg" : list(aligned_eeg_behav),
        "trigger_value" : beh_events_df["value"].values,
        "trial_count" : beh_events_df["trial_count"].values,
        "cumul_seq" : beh_events_df["cumul"].values,
        "value_log" : np.nan,
        "value_eeg" : np.nan,
        "time_signal" : np.nan,
        "time_log" : np.nan,
        "sample" : np.nan,
        "run" : np.nan
    })
    return events_df

        
def find_consecutive(signal):
    """
    Find all consecutive non-zero values in a binary signal
    """
    signal = np.array(signal)
    consecutive = []
    i = 0
    while i < len(signal) :
        if signal[i] == 1:
            idx_list = []
            while i < len(signal) and signal[i] == 1:
                idx_list.append(i)
                i += 1
            consecutive.append(idx_list)
        else:
            i += 1
    return consecutive


