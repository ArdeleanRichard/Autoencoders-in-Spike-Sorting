import os

import realdata_spikes as rds
import realdata_parsing as rd
import numpy as np

from utils.dataset_parsing.realdata_spikes import separate_by_channel, get_data_from_channel, find_waverform_files
import matplotlib.pyplot as plt

DATASET_PATH = '../../datasets/real_data/M017/'
WAVEFORM_LENGTH = 58
TIMESTAMP_LENGTH = 1
print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, _, _ = find_waverform_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")

spikes_per_channel = rds.parse_spktwe_file(DATASET_PATH)
print(f"Number of Channels: {spikes_per_channel.shape}")
print(f"Number of Spikes on all Channels: {np.sum(spikes_per_channel)}")
print("--------------------------------------------")

timestamps = rd.read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_channel)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_channel)}")
timestamps_by_channel = separate_by_channel(spikes_per_channel, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_channel}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_channel))}")
print(f"Assert equality: {list(spikes_per_channel) == list(map(len, timestamps_by_channel))}")
print("--------------------------------------------")

waveforms = rd.read_waveforms(waveform_file)
print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*58: {len(timestamps) * WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_channel = separate_by_channel(spikes_per_channel, waveforms, WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_channel))}")
print(f"Spikes per channel parsed from file: {spikes_per_channel}")
waveform_lens = list(map(len, waveforms_by_channel))
print(f"Waveforms/58 per channel should be equal: {[i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_channel) == [i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")

# timestamps by channel:
# get_data_from_channel(timestamps_by_channel, 2, length=TIMESTAMP_LENGTH)
# waveforms by channel:
# get_data_from_channel(waveforms_by_channel, 2, length=WAVEFORM_LENGTH)

# channels go from 1 to 32







# spike_units = rd.get_spike_units(waveforms, 1)
# print("spike_units read")
# print(spike_units)
# print(spike_units[0].shape)
# print(spike_units[1].shape)
# print()
# # ds.real_data_extract_spikes_specific_channel(timestamps_, waveforms_, channel=2)
#
# rd.plot_spikes_per_unit(waveforms)
#
# rd.fe_per_channels()