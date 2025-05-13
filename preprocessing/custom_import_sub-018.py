
#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Subject 18 custom import
"""

# Import libraries
from preprocessing.config import *
from preprocessing.utils.data_helper import *


file_path = os.path.join(ORIGINAL_DATA_DIR, "18", "EEG", "from_Anne")
fileslist = get_fileslist(file_path, ".edf")

file = fileslist[0]
raws = []
for file in fileslist:
    raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    new_type = ["seeg" if x == "eeg" else x for x in raw.get_channel_types()]
    mapping = {x: y for x, y in zip(raw.ch_names, new_type)}
    raw.set_channel_types(mapping)
    raws.append(raw)



full = mne.concatenate_raws(raws, preload=True)