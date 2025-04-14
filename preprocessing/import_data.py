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


