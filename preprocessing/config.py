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
import mne
from copy import deepcopy
import gc

# Set the path to directories
ORIGINAL_DIR = "/Users/charles.verstraete/Documents/w3_iEEG/"
ORIGINAL_DATA_DIR = os.path.join(ORIGINAL_DIR, "subject_collection")

FIRST_ANALYSIS__DATA_DIR = os.path.join(ORIGINAL_DIR, "analysis/data")

SECOND_ANALYSIS__DATA_DIR = os.path.join(ORIGINAL_DIR, "analysis_v2/data")

ROOT_DIR = os.path.join(ORIGINAL_DIR, "analysis_v3")
DATA_DIR = os.path.join(ROOT_DIR, "data")
PREPROCESSING_DIR = os.path.join(ROOT_DIR, "preprocessing")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

ATLAS_DIR = "/Users/charles.verstraete/Documents/w3_iEEG/anatomical_atlas"

# Set subjects and sessions
N_SUBJECTS = 28
BAD_SUBJECTS = [1, 6, 7, 10, 11, 13, 15, 17,18, 21, 22, 24, 26, 27]
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
    2 : ["r1", "h10", "or6", "oc10"],
    3 : [],
    4 : ["i5", "i12"],
    5 : [],
    8 : ["pam4"],
    9 : ["tsd10"],
    12 : ["ap3", "ap4", "ap5", "rp15"],
    14 : ["fmd18", "tmd12", "iig16", "hag12", "iig3"],
    16 : ["ofg15"],
    18 : [],
    19 : ["xpd12"],
    20 : ["g7", "g8"],
    23 : ["hpd14", "hpd15"],
    25 : ["gra4", "gra5"],
    28 : ["rsd15","fag1", "fag2", "fag3", "fag4", "fag5", "fag6", "fag7", "fag8", "fag9", "fag10"],
}

area_dict = {
    'Somatosensory' : 1,
    'Motor' : 2,
    'Premotor' : 3,
    'SMA' : 4,
    'preSMA' : 5,
    'Posterior_Insula' : 6,
    'Anterior_Insula' : 7,
    'FOP' : 8,
    'mid-VLPFC' : 9,
    'VLPFC_POST' : 10,
    'DLPFC_POST' : 11,
    'mid-DLPFC' : 12,
    'DMPFC' : 13,
    'MCCa' : 14,
    'ACC' : 15, 
    'VMPFC' : 16,
    'OFC' : 17,
    'Frontopolar' : 18
}


# area_colors = {
#     'Somatosensory': '#f17126',  # Orange plus intense
#     'Motor': '#f58640',          # Pêche-orangé plus saturé
#     'Premotor': '#f7a060',       # Pêche moyen renforcé
#     'SMA': '#f8b17b',            # Pêche clair avec plus de contraste
#     'preSMA': '#f9c298',         # Pêche clair, légèrement plus saturé
    
#     'Posterior_Insula': '#9de6b5',  # Vert menthe intensifié
#     'Anterior_Insula': '#82d69b',   # Vert plus saturé
#     'FOP': '#5ec783',              # Vert moyen plus vif
    
#     'mid-VLPFC': '#88c2e3',      # Bleu clair plus intense
#     'VLPFC_POST': '#5caad7',     # Bleu moyen plus profond
#     'DLPFC_POST': '#3992cb',     # Bleu plus dense et saturé
#     'mid-DLPFC': '#0d7ac4',      # Bleu profond plus marqué
    
#     'DMPFC': '#c594db',          # Lavande rosé plus vif
#     'MCCa': '#b475d0',           # Lavande moyen plus saturé
#     'ACC': '#a055c2',            # Violet plus intense
    
#     'VMPFC': '#f290c0',          # Rose plus saturé
#     'OFC': '#ec6ca9',            # Rose moyen plus vif
#     'Frontopolar': '#e54994'     # Rose-rouge plus profond
# }

area_colors = {
    # Gradient orange-pêche (espacement maximal)
    'Somatosensory': '#cc3300',  # Rouge-orange très foncé
    'Motor': '#e64d00',          # Orange foncé
    'Premotor': '#ff6600',       # Orange vif
    'SMA': '#ff8040',            # Orange moyen-clair (plus séparé)
    'preSMA': '#ffb380',         # Orange très clair (bien distinct)
    
    # Gradient vert (espacement maximal)
    'Posterior_Insula': '#1a5c2a',  # Vert très foncé
    'Anterior_Insula': '#2e7a40',   # Vert foncé
    'FOP': '#80cc99',              # Vert clair (très distinct)
    
    # Gradient bleu (espacement maximal)
    'mid-VLPFC': '#0d4d80',      # Bleu très foncé
    'VLPFC_POST': '#1a66a0',     # Bleu foncé
    'DLPFC_POST': '#4080cc',     # Bleu moyen-clair (plus séparé)
    'mid-DLPFC': '#80c0ff',      # Bleu clair (très distinct)
    
    # Gradient violet (espacement maximal)
    'DMPFC': '#4d1a66',          # Violet très foncé
    'MCCa': '#7040aa',           # Violet moyen-clair (plus séparé)
    'ACC': '#b380e6',            # Violet clair (très distinct)
    
    # Gradient rose (espacement maximal)
    'VMPFC': '#990033',          # Rose très foncé
    'OFC': '#cc3366',            # Rose moyen
    'Frontopolar': '#ff80b3'     # Rose clair (très distinct)
}

palette = sns.color_palette("Dark2")
stable_color = palette[3]
partial_color = palette[2]
complete_color = palette[0]
palette_dict = {0: partial_color, 1: stable_color, -1: complete_color, 
                'random': "grey", 'global': complete_color, 'overlap': partial_color}

stim_ids = [10, 20, 30, 11, 21, 31, 12, 22, 32]

epoch_padding = 6
decimation = 8

n_freqs = 80

total_duration = 2 * epoch_padding 
n_times = int(SAMPLING_RATE * total_duration) + 1
times = np.linspace(-epoch_padding, epoch_padding, n_times, endpoint=True)
times_decimed = times[::decimation]
n_times_decimed = len(times_decimed)

sr_decimated = int(SAMPLING_RATE/decimation)
freqlist = np.logspace(np.log10(2), np.log10(200), n_freqs)
freqlist = np.round(freqlist*total_duration)/total_duration
cycles = np.where(freqlist <= 32, 6.0, np.floor(0.24 * freqlist))

frequency_bands = {
    'delta': (2, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'low_beta': (12, 20),
    'high_beta': (20, 40),
    'low_gamma': (40, 80),
    'high_gamma': (80, 200)
}

fr_cutoff = [value for _, value in frequency_bands.items()]
band_indices = [np.argmin(np.abs(freqlist - val[0])) for val in fr_cutoff]
band_indices.append(len(freqlist)-1)
n_frband = len(frequency_bands)


WOI_start_time = 1.5
WOI_end_time = 3.5
WOI_start_idx= int((epoch_padding - WOI_start_time) * sr_decimated)
WOI_end_idx = int((WOI_end_time + epoch_padding) * sr_decimated)
WOI_onset_idx = int((epoch_padding) * sr_decimated)


