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
def add_before_pres(df) :
    """
    Add the before presentation column to the dataframe
    """
    df["before_pres"] = np.nan
    df["next_stable"] = np.nan
    stim_count = np.zeros(3)
    for idx, row in df.iterrows() :
        if row["new_block"] == 1 :
            if idx > 1 :                   
                stim_count = np.zeros(3)
                who_stable = row["who_stable"]
                t = idx
                while stim_count[0] < 6 and stim_count[1] < 6 and stim_count[2] < 6 and t > 0 :
                    t -= 1
                    stim_count[df.loc[t, "stim"]-1] += 1
                    df.loc[t, "before_pres"] = -stim_count[df.loc[t, "stim"]-1]
                    if np.isnan(who_stable) :
                        df.loc[t, "next_stable"] = -1
                    else :
                        if df.loc[t, "stim"] == who_stable :
                            df.loc[t, "next_stable"] = 1
                        else :
                            df.loc[t, "next_stable"] = 0
    df["before_pres"] = beh_df["before_pres"].fillna(0)
    return df

def add_before_trial(df) :
    """
    Add the before presentation column to the dataframe
    """
    df["before_trial"] = np.nan
    trial_count = 0
    for idx, row in df.iterrows() :
        if row["new_block"] == 1 :
            if idx > 1 :              
                trial_count = 0     
                who_stable = row["who_stable"]
                t = idx
                while trial_count < 16 and t > 0 :
                    t -= 1
                    trial_count += 1
                    df.loc[t, "before_trial"] = -trial_count
    df["before_trial"] = beh_df["before_trial"].fillna(0)
    return df

def find_criterion(df) :
    """
    Find the criterion for the before presentation column
    """
    df["criterion"] = False
    good_count = 0
    stim_count = np.zeros(3)
    for idx, row in df.iterrows() :
        if row["new_block"] == 1 :
            good_count = 0
            stim_count = np.zeros(3)
        if row["trial"] > 15 : 
            good_count = df.loc[idx-5: idx, "correct"].sum()
            stim_count[row["stim"] - 1] = (stim_count[row["stim"] - 1] + row["correct"])*row["correct"]
        if ((good_count >= 4) & (np.sum(stim_count > 1) == 3)) :
            df.loc[idx-6, "criterion"] = True
            df.loc[idx-7, "criterion"] = True
    return df
