#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Configuration file for global variables and constants
"""

# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the path to directories
ORIGINAL_DIR = "/Users/charles.verstraete/Documents/w3_iEEG/"
ORIGINAL_DATA_DIR = os.path.join(ORIGINAL_DIR, "subject_collection")

FIRST_ANALYSIS__DATA_DIR = os.path.join(ORIGINAL_DIR, "analysis/data")
electrode_first_extensions = ["raw", "anatomical/implantation/"]

SECOND_ANALYSIS__DATA_DIR = os.path.join(ORIGINAL_DIR, "analysis_v2/data")

ROOT_DIR = "/Users/charles.verstraete/Documents/w3_iEEG/analysis_v3/"
DATA_DIR = os.path.join(ROOT_DIR, "data")
PREPROCESSING_DIR = os.path.join(ROOT_DIR, "preprocessing")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# Set subjects and sessions
N_SUBJECTS = 25
BAD_SUBJECTS = [1, 6, 7, 8, 10, 11, 12, 13, 15, 17, 21, 22, 23, 24, 25]
SUBJECTS = [i for i in range(1, N_SUBJECTS + 1) if i not in BAD_SUBJECTS]
SUBJECTS_NAME = [f'sub-{i:03d}' for i in range(1, N_SUBJECTS + 1) if i not in BAD_SUBJECTS]

# Set constants
LINENOISE = 50
SAMPLING_RATE = 2048
HIGHPASS = 2
LOWPASS = None

FREQUENCY_BANDS = {
    'delta': (2, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'low_beta': (12, 20),
    'high_beta': (20, 40),
    'low_gamma': (40, 80),
    'high_gamma': (80, 150)
}

BAD_CHANNELS = {
    1 : [], 
    2 : ["r1", "oc10"],
    3 : [],
    4 : ["i5"],
    5 : [],
    6 : [],
    7 : [],
    8 : [],
    9 : [],
    10 : [],
    11 : [],
    12 : [],
    13 : [],
    14 : [],
    15 : [],
    16 : ["ofg15"],
    17 : [],
    18 : [],
    19 : [],
    20 : ["o6", "o7", "o8", "f9"],
    21 : [],
    22 : [],
    23 : [],
    24 : [],
    25 : [],
    26 : [],
    27 : [],
    28 : [],
    29 : [],
    30 : [],
}




palette = sns.color_palette("Dark2")
stable_color = palette[3]
partial_color = palette[2]
complete_color = palette[0]
palette_dict = {0: partial_color, 1: stable_color, -1: complete_color}







