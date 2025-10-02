#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Helper functions for behavioral data 
"""
# Import libraries

import numpy as np
import pandas as pd 
import mne

from preprocessing.config import *

# Functions definitions
# def add_before_pres(df) :
#     """
#     Add the before presentation column to the dataframe
#     """
#     df["before_pres"] = np.nan
#     df["next_stable"] = np.nan
#     stim_count = np.zeros(3)
#     for idx, row in df.iterrows() :
#         if row["new_block"] == 1 :
#             if idx > 1 :                   
#                 stim_count = np.zeros(3)
#                 who_stable = row["who_stable"]
#                 t = idx
#                 while stim_count[0] < 6 and stim_count[1] < 6 and stim_count[2] < 6 and t > 0 :
#                     t -= 1
#                     stim_count[df.loc[t, "stim"]-1] += 1
#                     df.loc[t, "before_pres"] = -stim_count[df.loc[t, "stim"]-1]
#                     if np.isnan(who_stable) :
#                         df.loc[t, "next_stable"] = -1
#                     else :
#                         if df.loc[t, "stim"] == who_stable :
#                             df.loc[t, "next_stable"] = 1
#                         else :
#                             df.loc[t, "next_stable"] = 0
#     df["before_pres"] = df["before_pres"].fillna(0)
#     return df

# def add_before_trial(df) :
#     """
#     Add the before presentation column to the dataframe
#     """
#     df["before_trial"] = np.nan
#     trial_count = 0
#     for idx, row in df.iterrows() :
#         if row["new_block"] == 1 :
#             if idx > 1 :              
#                 trial_count = 0     
#                 who_stable = row["who_stable"]
#                 t = idx
#                 while trial_count < 16 and t > 0 :
#                     t -= 1
#                     trial_count += 1
#                     df.loc[t, "before_trial"] = -trial_count
#     df["before_trial"] = df["before_trial"].fillna(0)
#     return df

# def find_criterion(df) :
#     """
#     Find the criterion for the before presentation column
#     """
#     df["criterion"] = False
#     good_count = 0
#     stim_count = np.zeros(3)
#     for idx, row in df.iterrows() :
#         if row["new_block"] == 1 :
#             good_count = 0
#             stim_count = np.zeros(3)
#         if row["trial"] > 15 : 
#             good_count = df.loc[idx-5: idx, "correct"].sum()
#             stim_count[row["stim"] - 1] = (stim_count[row["stim"] - 1] + row["correct"])*row["correct"]
#         if ((good_count >= 4) & (np.sum(stim_count > 1) == 3)) :
#             df.loc[idx-6, "criterion"] = True
#             # df.loc[idx-7, "criterion"] = True
#     return df



def zscore(vect):
    return (vect-np.mean(vect))/np.std(vect)


def get_df(fileslist, rt_zscore=False, filter=False, add_prev=False):
    df_list = []
    for file in fileslist:
        tmp_df = pd.read_csv(file)
        # tmp_df["trial_count"] = np.arange(1, len(tmp_df) + 1, dtype=int)
        # tmp_df = add_persev_explor(tmp_df)
        tmp_df = find_criterion(tmp_df)
        tmp_df = tmp_df[tmp_df["criterion"] == 0].reset_index(drop=True)
        if rt_zscore and "rt" in tmp_df.columns:
            tmp_df = tmp_df[(tmp_df["rt"] > 0) & (tmp_df["rt"] < 5)]
            tmp_df = tmp_df.reset_index(drop=True)
            tmp_df["rt_zscore"] = zscore(tmp_df["rt"].values)
        if filter:
            tmp_df.loc[tmp_df["is_stimstable"].isna(), "is_stimstable"] = -1
            tmp_df["fb_prev"] = 0
            tmp_df = tmp_df[tmp_df["training"] == 0]
            tmp_df = tmp_df[tmp_df["miss"] == 0]
            tmp_df = tmp_df.reset_index(drop=True)
            tmp_df.loc[1:, "fb_prev"] = tmp_df["fb"].values[:-1]
        if add_prev:
            tmp_df = add_before_pres(tmp_df)
            tmp_df = add_before_trial(tmp_df)
            tmp_df = tmp_df.reset_index(drop=True)
        tmp_df = recount_trials(tmp_df)
        tmp_df = recount_trial_switch(tmp_df)
        df_list.append(tmp_df)

    df = pd.concat(df_list, ignore_index=True)
    return df

def recount_trial_switch(df):
    df = df.copy().reset_index(drop=True)
    trials = []
    stim_pres_count = np.zeros(3, dtype=int)
    stim_pres = []
    first_switch = False
    for _, row in df.iterrows():
        if not first_switch:
            first_switch = row["firstswitch"] == 1
        if first_switch :
            if row["firstswitch"] == 1 :
                stim_pres_count[:] = 0
                trials.append(1)
                stim_pres_count[int(row["stim"]) - 1] += 1
                stim_pres.append(stim_pres_count[int(row["stim"]) - 1])
            else:
                trials.append(trials[-1] + 1)
                stim_pres_count[int(row["stim"]) - 1] += 1
                stim_pres.append(stim_pres_count[int(row["stim"]) - 1])
        else :
            trials.append(0)
            stim_pres.append(0)
    
    df["post_hmmsw_trial"] = trials
    df["post_hmmsw_pres"] = stim_pres
    return df

def recount_trials(df):
    df = df.copy().reset_index(drop=True)
    trials = []
    stim_pres_count = np.zeros(3, dtype=int)
    stim_pres = []
    for _, row in df.iterrows():
        if row["new_block"] == 1 or row["trial"] == 1:
            stim_pres_count[:] = 0
            trials.append(1)
            stim_pres_count[int(row["stim"]) - 1] += 1
            stim_pres.append(stim_pres_count[int(row["stim"]) - 1])
        else:
            trials.append(trials[-1] + 1)
            stim_pres_count[int(row["stim"]) - 1] += 1
            stim_pres.append(stim_pres_count[int(row["stim"]) - 1])
    df["trial"] = trials
    df["stim_pres"] = stim_pres
    return df

def add_before_pres(df) :
    """
    Add the before presentation column to the dataframe
    - Utilise un index positionnel (RangeIndex) pour éviter les KeyError.
    """
    df = df.copy().reset_index(drop=True)  # IMPORTANT: index positionnel
    df["before_pres"] = np.nan
    df["next_stable"] = np.nan
    stim_count = np.zeros(3, dtype=int)
    for i, row in df.iterrows() :
        if row["new_block"] == 1 :
            if i > 1 :
                stim_count[:] = 0
                who_stable = row.get("who_stable", np.nan)
                t = i
                # on remonte tant que chaque stimulus a moins de 6 présences
                while (stim_count < 6).all() and t > 0 :
                    t -= 1
                    stim_idx = int(df.at[t, "stim"]) - 1
                    stim_count[stim_idx] += 1
                    df.at[t, "before_pres"] = -stim_count[stim_idx]
                    if pd.isna(who_stable):
                        df.at[t, "next_stable"] = -1
                    else:
                        df.at[t, "next_stable"] = 1 if int(df.at[t, "stim"]) == int(who_stable) else 0
    df["before_pres"] = df["before_pres"].fillna(0)
    return df

def add_before_trial(df) :
    """
    Add the before trial column to the dataframe
    - Utilise un index positionnel (RangeIndex) pour éviter les KeyError.
    """
    df = df.copy().reset_index(drop=True)  # IMPORTANT: index positionnel
    df["before_trial"] = np.nan
    trial_count = 0
    for i, row in df.iterrows() :
        if row["new_block"] == 1 :
            if i > 1 :
                trial_count = 0
                t = i
                while trial_count < 16 and t > 0 :
                    t -= 1
                    trial_count += 1
                    df.at[t, "before_trial"] = -trial_count
    df["before_trial"] = df["before_trial"].fillna(0)
    return df

def find_criterion(df) :
    """
    Find the criterion for the before presentation column
    - Sécurisé pour index positionnel.
    """
    df = df.copy().reset_index(drop=True)  
    df["criterion"] = False
    df["post_criterion"] = 0
    df["finished"] = False

    n = len(df)

    corrects = df["correct"].astype(int).values
    good_count = [0, 0, 0]  
    success = 0             
    post_perf = 0           
    finished_episode = False
    t_epis = 0 

    for i in range(n):
        if (df.at[i, "new_block"] == 1) or (df.at[i, "trial"] == 1):
            good_count = [0, 0, 0]
            success = 0
            post_perf = 0
            finished_episode = False
            t_epis = 0

        if t_epis > 8:
            c = corrects[i]
            s = int(df.at[i, "stim"]) - 1
            success = sum(corrects[t_epis-8:t_epis])
            good_count[s] = (good_count[s] + c) * c
        
        perf_criterion = (success >= 5) and (sum(gc > 1 for gc in good_count) == 3)
        if perf_criterion or post_perf > 0:
            post_perf += 1

        finished_episode = post_perf > 5

        # Marque criterion au premier essai post-critère
        if i > 1 and (post_perf == 1):
            df.at[i - 1, "criterion"] = df.at[i, "criterion"]

        df.at[i, "post_criterion"] = int(post_perf)
        df.at[i, "finished"] = bool(finished_episode)

        t_epis += 1

    return df

def add_persev_explor(df) :
    past_rule = [0, 0, 0]
    pervev_choices = []
    explor_choices = []
    df = df.copy().reset_index(drop=True)
    df["stim"] = df["stim"].astype(int)

    for idx, row in df.iterrows() :
        if row["epis"] == 0 :
            pervev_choices.append(0)
            explor_choices.append(0)
        else :
            if row["new_block"] == 1 :
                past_rule = [df.at[idx-1, "rule_1"], df.at[idx-1, "rule_2"], df.at[idx-1,"rule_3"]]
            s = int(row["stim"])
            pervev_choices.append(past_rule[s - 1])
            candidates = {1, 2, 3}
            remove = set([past_rule[s - 1], row["correct_choice"]])
            explor_choices.append(list(candidates - remove)[0])
    df["persev_choice"] = pervev_choices
    df["persev"] = (df["persev_choice"] == df["choice"])
    df["explor_choice"] = explor_choices
    df["explor"] = (df["explor_choice"] == df["choice"])
    return df


def filter_episodes(epis_count, min_count=15, max_count=55):
    """
        Filter episodes based on min and max count
    """
    mask = (epis_count["count"] > min_count) & (epis_count["count"] < max_count)

    filtered = epis_count.loc[mask].copy().reset_index(drop=True)
    removed = epis_count.loc[~mask].copy().reset_index(drop=True)

    totals = (
        epis_count.groupby("subject")
        .size()
        .rename("n_total")
        .reset_index()
    )
    removed_counts = (
        removed.groupby("subject")
        .size()
        .rename("removed_count")
        .reset_index()
    )

    synthese_removed = totals.merge(removed_counts, on="subject", how="left")
    synthese_removed["removed_count"] = synthese_removed["removed_count"].fillna(0).astype(int)
    synthese_removed["kept_count"] = synthese_removed["n_total"] - synthese_removed["removed_count"]
    synthese_removed["proportion"] = synthese_removed["removed_count"] / synthese_removed["n_total"]

    return filtered, removed, synthese_removed




def mean_sem(df, x, y, hue=None, within="subject"):
    """
    Agrège en moyenne/sem par x (et hue si fourni).
    - within: si présent dans df, calcule d'abord la moyenne par within, puis moyenne/sem entre within.
    """
    cols = [x] + ([hue] if hue else [])
    if within is not None :
        indiv = df.groupby([within] + cols, dropna=False)[y].mean().reset_index()
        summ = indiv.groupby(cols, dropna=False)[y].agg(["mean", "std", "count"]).reset_index()
    else:
        summ = df.groupby(cols, dropna=False)[y].agg(["mean", "std", "count"]).reset_index()
    summ["sem"] = summ["std"] / np.sqrt(summ["count"])
    return summ


def get_switch_summary(df, x_pre, x_post, y_var, pre_lim, post_lim, hue_pre = None, hue_post = None, within="subject"):
    """
    Get summary statistics around switch points
    """

    summary_after = mean_sem(df, x_post, y_var, hue=hue_post, within=within)
    summary_before = mean_sem(df, x_pre, y_var, hue=hue_pre, within=within)

    summary_after = summary_after[(summary_after[x_post] > 0) & (summary_after[x_post] < post_lim)]
    summary_before = summary_before[(summary_before[x_pre] < 0) & (summary_before[x_pre] > pre_lim)]

    return summary_before, summary_after


