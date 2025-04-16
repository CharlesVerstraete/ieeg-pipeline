#-*- coding: utf-8 -*-
#-*- python 3.9.6 -*-

# Author : Charles Verstraete
# Date : 2025

""" 
Helper functions for data import and manipulation
"""

# Import libraries

import numpy as np
import pandas as pd
import quantities as pq

import neo
import mne
import nibabel as nib

import matplotlib.pyplot as plt

from preprocessing.config import *

# Functions definitions

def create_subject_folder(subject):
    """
    Create a folder for the subject and subfolders for : 
    - raw data (ieeg, anat, beh)
    - preprocessed data (aligned, filtered, epochs, timefreq)
    """
    # Create the subject folder
    subject_folder = os.path.join(DATA_DIR, f'sub-{subject:03d}')
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    # Create subfolders for raw data
    raw_data_folder = os.path.join(subject_folder, 'raw')
    if not os.path.exists(raw_data_folder):
        os.makedirs(raw_data_folder)
        os.makedirs(os.path.join(raw_data_folder, 'ieeg'))
        os.makedirs(os.path.join(raw_data_folder, 'anat'))
        os.makedirs(os.path.join(raw_data_folder, 'beh'))

    # Create subfolders for preprocessed data
    preprocessed_data_folder = os.path.join(subject_folder, 'preprocessed')
    if not os.path.exists(preprocessed_data_folder):
        os.makedirs(preprocessed_data_folder)
        os.makedirs(os.path.join(preprocessed_data_folder, 'aligned'))
        os.makedirs(os.path.join(preprocessed_data_folder, 'filtered'))
        os.makedirs(os.path.join(preprocessed_data_folder, 'epochs'))
        os.makedirs(os.path.join(preprocessed_data_folder, 'timefreq'))

def check_extensions(filelist) : 
    """
    Check the extensions of the files in the directory
    """
    # Get the list of files in the directory    
    extensions = [os.path.splitext(f)[1].lower() for f in filelist]
    unique_extensions = set(extensions)
    return unique_extensions

def get_fileslist(path, extension):
    """
    Return a sorted list of files in 'path' with the specified 'extension'.
    """
    return sorted(os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(extension))

def extract_trc(file_path):
    """
    Extract data from a .TRC file using Neo's MicromedIO and return a segment object.
    """
    io = neo.MicromedIO(filename=file_path)
    segment = io.read_segment()
    segment.annotate(material="micromed")
    del io 
    return segment

def format_channelname(segment):
    """
    Format the channel names by removing spaces and replacing apostrophes with 'p'.
    """
    for asignal in segment.analogsignals:
        n_channels = len(asignal.array_annotations["channel_names"])
        for i in range(n_channels) :
            channel = asignal.array_annotations["channel_names"][i]
            if " " in channel:
                asignal.array_annotations["channel_names"][i] = channel.replace(" ", "")
            if "'" in channel:
                asignal.array_annotations["channel_names"][i] = channel.replace("'", "p")
    return segment


def check_units(segment):
    """
    Check the units of the signals and rescale them if necessary.
    """
    n = len(segment.analogsignals)
    for i in range(n):
        if segment.analogsignals[i].units.dimensionality.string == "uV":
            segment.analogsignals[i] = segment.analogsignals[i].rescale(pq.V)
            print("\n #### Data rescaled from uV to V\n")
    return segment

def get_anat_channelnames(subject) :
    """
    Get the channel names from the anat file
    """
    electrode_path = os.path.join(FIRST_ANALYSIS__DATA_DIR, electrode_first_extensions[0], f'subject_{subject:02d}', electrode_first_extensions[1])
    electrodes_files = get_fileslist(electrode_path, ".txt")
    with open(electrodes_files[0], "r") as ename_file:
        channel_names = [line.strip().lower() for line in ename_file]
    return channel_names

def find_electrodes_types(channel_signal, channel_anat, patterns = ["ecg", "eog", "emg"]):
    """
    Classify the electrodes into different types based on their names.
    """
    results = []
    for i, name in enumerate(channel_signal):
        name_lower = name.lower()
        if name_lower in channel_anat:
            results.append("seeg")
        else:
            for type_name in patterns:
                if type_name.lower() in name_lower:
                    results.append(type_name)
                    break
            else: 
                results.append("bio")
    return results

def events_fromsignal(segment):
    """
    Create events from the signal
    """
    time = segment._events[0].times.magnitude
    sfreq = segment.analogsignals[0].sampling_rate
    label = segment._events[0].labels.astype(int)
    return np.vstack(((time*sfreq).magnitude, np.zeros(len(time)), label)).astype(int).T

def import_trc_file(ieeg_file, subject = None):
    """
    Import the ieeg file and return the signal, channel names and electrodes types
    """
    segment = extract_trc(ieeg_file)
    segment = check_units(segment)
    segment = format_channelname(segment)
    if subject is not None:
        channel_anat = get_anat_channelnames(subject)
    else:
        channel_anat = []
    channel_signal = []
    for (i, asignal) in enumerate(segment.analogsignals):
        channel_signal = np.append(channel_signal, asignal.array_annotations["channel_names"])
    electrodes_types = find_electrodes_types(channel_signal, channel_anat)
    signal = np.concatenate([asignal.magnitude for asignal in segment.analogsignals], axis = 1)
    events = events_fromsignal(segment)
    sfreq = segment.analogsignals[0].sampling_rate.magnitude.astype(int)
    del segment, channel_anat
    return signal, channel_signal, electrodes_types, events, sfreq

def add_stim_events(raw, events):
    """
    Add stim events to the raw object
    """
    stim_data = np.zeros((1, len(raw.times)))
    stim_info = mne.create_info(['STIM'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, stim_info)
    raw.add_channels([stim_raw], force_update_info=True)
    raw.add_events(np.array(events), stim_channel='STIM')
    return raw

def create_mne(signal, channel_signal, channel_types, sampling_rate, events):
    """
    Create a MNE Raw object from the signal and channel names
    """
    info = mne.create_info(ch_names = channel_signal.tolist(), 
                           sfreq = sampling_rate, 
                           ch_types = channel_types)
    raw = mne.io.RawArray(signal.T, info, verbose = False)
    raw = add_stim_events(raw, events)
    if raw.info['sfreq'] < SAMPLING_RATE:
        raw, events = raw.resample(SAMPLING_RATE, events = events, n_jobs=-1)
    return raw, events

def collect_raw_data(subject):
    """
    Collect raw data from the original dataset and save it in BIDS-like format
    """
    raw_data_folder = os.path.join(ORIGINAL_DATA_DIR, str(subject), 'EEG')
    raw_files = os.listdir(raw_data_folder)
    unique_extensions = check_extensions(raw_files)
    if '.trc' in unique_extensions :
        ieeg_files = get_fileslist(raw_data_folder, '.trc')
        print(f"\n#### Subject {subject}")
        for (i, ieeg_file) in enumerate(ieeg_files):
            idx = i + 1
            signal, channel_signal, electrodes_types, events, sfreq = import_trc_file(ieeg_file, subject)
            raw, events = create_mne(signal, channel_signal, electrodes_types, sfreq, events)
            save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}',  'raw', 'ieeg', f'sub-{subject:03d}_run-{idx:02d}-raw.fif')
            raw.save(save_path, overwrite=True)
            print(" "*10 + f"Saving {ieeg_file.split('/')[-1]} as sub-{subject:03d}_run-{idx:02d}-raw.fif")
            del raw, events, signal, channel_signal, electrodes_types
    # elif '.edf' in unique_extensions :
    #     ieeg_files = get_fileslist(raw_data_folder, '.edf')
    #     for (i, ieeg_file) in enumerate(ieeg_files):
    #         idx = i + 1
    #         print(f"\n#### Subject {subject}")
    #         raw = mne.io.read_raw_edf(ieeg_file, preload=True)
    #         events = mne.find_events(raw)
    #         if raw.info['sfreq'] < SAMPLING_RATE:
    #             raw, events = raw.resample(SAMPLING_RATE, events = events, n_jobs=-1)
    #         save_path = os.path.join(DATA_DIR, f'sub-{subject:03d}', 'raw', 'ieeg', f'sub-{subject:03d}_run-{idx:02d}-raw.fif')
    #         raw.save(save_path, overwrite=True)
    #         print(" "*10 + f"Saving {ieeg_file.split('/')[-1]} as sub-{subject:03d}_run-{idx:02d}-raw.fif")
    #         del raw, events

        
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


