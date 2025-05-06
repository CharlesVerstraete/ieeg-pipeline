
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

ORIGINAL_DATA_DIR
file_path = os.path.join(ORIGINAL_DATA_DIR, "18", "EEG", "ncs")
fileslist = get_fileslist(file_path, ".ncs")


raw = mne.io.read_raw_neuralynx(file_path)

nlx_reader = neo.NeuralynxIO(dirname=file_path) 





nlx_reader.header["signal_channels"]["name"].tolist()
nlx_reader.get_signal_sampling_rate()
nlx_reader.parse_header()
print(f"Channel info via Neo: {nlx_reader.header['nb_channel']} channels found")
test = nlx_reader.get_analogsignal_chunk(block_index=0, seg_index=100)



ncs_fnames = nlx_reader.ncs_filenames.values()
ncs_hdrs = [
    hdr
    for hdr_key, hdr in nlx_reader.file_headers.items()
    if hdr_key in ncs_fnames
]


# Neo reads only valid contiguous .ncs samples grouped as segments
n_segments = nlx_reader.header["nb_segment"][0]
block_id = 0  # assumes there's only one block of recording

# get segment start/stop times
start_times = np.array(
    [nlx_reader.segment_t_start(block_id, i) for i in range(n_segments)]
)
stop_times = np.array(
    [nlx_reader.segment_t_stop(block_id, i) for i in range(n_segments)]
)

next_start_times = start_times[1::]
previous_stop_times = stop_times[:-1]
seg_diffs = next_start_times - previous_stop_times
delta = 1.5 / nlx_reader.get_signal_sampling_rate()
gaps = seg_diffs > delta

seg_gap_dict = {}

gap_starts = stop_times[:-1][gaps]  
gap_stops = start_times[1::][gaps]  

gap_n_samps = np.array(
    [
        int(round(stop * nlx_reader.get_signal_sampling_rate())) - int(round(start * nlx_reader.get_signal_sampling_rate()))
        for start, stop in zip(gap_starts, gap_stops)
    ]
).astype(int)  # force an int array (if no gaps, empty array is a float)

nlx_reader.get_signal_size(block_id, )





file.read_segment()
segment = file.read_segment()

segment._events