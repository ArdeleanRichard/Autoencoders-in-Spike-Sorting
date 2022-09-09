from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import realdata_spikes as rds
import realdata_parsing as rd
from utils.dataset_parsing.realdata_spikes import parse_spktwe_file, find_waverform_files
from utils.dataset_parsing.realdata_ssd import plot_sorted_data

DATASET_PATH = '../../datasets/kampff/c37/Waveforms/'

WAVEFORM_LENGTH = 54
TIMESTAMP_LENGTH = 1
print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_waverform_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")


spikes_per_channel = rds.parse_spktwe_file(DATASET_PATH, NR_CHANNELS=1)
print(f"Number of Channels: {spikes_per_channel.shape}")
print(f"Number of Spikes on all Channels: {np.sum(spikes_per_channel)}")
print("--------------------------------------------")


timestamps = rd.read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_channel)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_channel)}")
timestamps_by_channel = rds.separate_by_channel(spikes_per_channel, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_channel}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_channel))}")
print(f"Assert equality: {list(spikes_per_channel) == list(map(len, timestamps_by_channel))}")
print("--------------------------------------------")

waveforms = rd.read_waveforms(waveform_file)
print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*58: {len(timestamps) * WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_channel = rds.separate_by_channel(spikes_per_channel, waveforms, WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_channel))}")
print(f"Spikes per channel parsed from file: {spikes_per_channel}")
waveform_lens = list(map(len, waveforms_by_channel))
print(f"Waveforms/58 per channel should be equal: {[i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_channel) == [i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")

event_timestamps = rd.read_event_timestamps(event_timestamps_filename)
print(f"Event Timestamps found in file: {event_timestamps.shape}")
event_codes = rd.read_event_codes(event_codes_filename)
print(f"Event Codes found in file: {event_codes.shape}")
# print(event_timestamps)
# print(event_codes)
print(f"Assert equality: {list(event_timestamps) == len(event_codes)}")
print("--------------------------------------------")


# print(waveforms_by_channel.shape)
# print(waveforms.shape)
waveforms_split = waveforms_by_channel.reshape((-1, WAVEFORM_LENGTH))
# print(waveforms_split.shape)


plot_sorted_data("", waveforms_split, None, nr_dim=2, show=True)

intracellular_labels = np.zeros((len(timestamps)))
print(len(intracellular_labels))
# given_index = np.zeros((len(event_timestamps[event_codes == 1])))
# for index, timestamp in enumerate(timestamps):
#     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
#         if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH and given_index[index2] == 0:
#             # given_index[index2] = 1
#             intracellular_labels[index] = 1
#             break


for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
    indexes = []
    for index, timestamp in enumerate(timestamps):
        if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH:
            # given_index[index2] = 1
            indexes.append(index)

    if indexes != []:
        min = indexes[0]
        for i in range(1, len(indexes)):
            if timestamps[indexes[i]] < timestamps[min]:
                min = indexes[i]
        intracellular_labels[min] = 1


print(len(intracellular_labels))
print(np.count_nonzero(np.array(intracellular_labels)))

plot_sorted_data("", waveforms_split, intracellular_labels, nr_dim=2, show=True)
plot_sorted_data("", waveforms_split, intracellular_labels, nr_dim=3, show=True)