import numpy as np

from dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, read_event_codes
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel, plot_sorted_data

DATASET_PATH = '../../datasets/kampff/c28/units/'

spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
WAVEFORM_LENGTH = 54
TIMESTAMP_LENGTH = 1
NR_CHANNELS = 1
unit_electrode = [1,1,1,1]

print(f"Number of Units: {spikes_per_unit.shape}")
print(f"Number of Units: {len(unit_electrode)}")
print(f"Number of Spikes in all Units: {np.sum(spikes_per_unit)}")
print(f"Unit - Electrode Assignment: {unit_electrode}")
print("--------------------------------------------")

print(f"DATASET is in folder: {DATASET_PATH}")
timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
print(f"TIMESTAMP file found: {timestamp_file}")
print(f"WAVEFORM file found: {waveform_file}")
print("--------------------------------------------")

timestamps = read_timestamps(timestamp_file)
print(f"Timestamps found in file: {timestamps.shape}")
print(f"Number of spikes in all channels should be equal: {np.sum(spikes_per_unit)}")
print(f"Assert equality: {len(timestamps) == np.sum(spikes_per_unit)}")

timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
print(f"Timestamps per channel should be equal: {list(map(len, timestamps_by_unit))}")
print(f"Assert equality: {list(spikes_per_unit) == list(map(len, timestamps_by_unit))}")
print("--------------------------------------------")

waveforms = read_waveforms(waveform_file)
print(f"Waveforms found in file: {waveforms.shape}")
print(f"Waveforms should be Timestamps*{WAVEFORM_LENGTH}: {len(timestamps) * WAVEFORM_LENGTH}")
print(f"Assert equality: {len(timestamps) * WAVEFORM_LENGTH == len(waveforms)}")
waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
print(f"Waveforms per channel: {list(map(len, waveforms_by_unit))}")
print(f"Spikes per channel parsed from file: {spikes_per_unit}")
waveform_lens = list(map(len, waveforms_by_unit))
print(f"Waveforms/{WAVEFORM_LENGTH} per channel should be equal: {[i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Assert equality: {list(spikes_per_unit) == [i//WAVEFORM_LENGTH for i in waveform_lens]}")
print(f"Sum of lengths equal to total: {len(waveforms) == np.sum(np.array(waveform_lens))}")
print("--------------------------------------------")

event_timestamps = read_event_timestamps(event_timestamps_filename)
print(f"Event Timestamps found in file: {event_timestamps.shape}")
event_codes = read_event_codes(event_codes_filename)
print(f"Event Codes found in file: {event_codes.shape}")
# print(event_timestamps)
# print(event_codes)
print(f"Assert equality: {list(event_timestamps) == len(event_codes)}")
print("--------------------------------------------")

print(event_timestamps, len(event_timestamps))
print(event_codes, len(event_codes))
print(event_timestamps[event_codes == 1])
print(timestamps, len(timestamps))

print(waveforms_by_unit.shape)
print(waveforms_by_unit[0].shape)

# plot_spikes_on_unit(waveforms_by_unit, 0, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 1, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 2, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)
# plot_spikes_on_unit(waveforms_by_unit, 3, WAVEFORM_LENGTH=WAVEFORM_LENGTH, show=True)

units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=NR_CHANNELS)

plot_sorted_data("", units_in_channels[0], labels[0], nr_dim=2, show=True)

intracellular_labels = np.zeros((len(timestamps)))
# given_index = np.zeros((len(event_timestamps[event_codes == 1])))
# for index, timestamp in enumerate(timestamps):
#     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
#         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
#             given_index[index2] = 1
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

plot_sorted_data("", units_in_channels[0], labels[0], nr_dim=2, show=True)
plot_sorted_data("", units_in_channels[0], intracellular_labels, nr_dim=2, show=True)
plot_sorted_data("", units_in_channels[0], labels[0], nr_dim=3, show=True)
plot_sorted_data("", units_in_channels[0], intracellular_labels, nr_dim=3, show=True)





# check amplitude scaling for intracellular on pca - unfortunately, its okay
# def plot_spikes(data, show=True):
#     plt.figure()
#     plt.title(f"Spikes ({len(data)})")
#     for i in range(0, len(data)):
#         plt.plot(np.arange(len(data[i])), data[i])
#
#     if show:
#         plt.show()
#
# intracellular_labels = np.array(intracellular_labels)
# spikes = np.array(units_in_channels[0])
#
#
# plot_spikes(spikes[intracellular_labels==1])
# plot_spikes(spikes[intracellular_labels==0])
#
# spikes[intracellular_labels==1] = spike_scaling_min_max(spikes[intracellular_labels==1],
#                                min_peak=np.amin(spikes[intracellular_labels==0]),
#                                max_peak=np.amax(spikes[intracellular_labels==0]))
#
# plot_spikes(spikes[intracellular_labels==1])
# plot_spikes(spikes[intracellular_labels==0])
#
# plot_sorted_data("", spikes, labels[0], nr_dim=2, show=True)
# plot_sorted_data("", spikes, intracellular_labels, nr_dim=2, show=True)
# plot_sorted_data("", spikes, labels[0], nr_dim=3, show=True)
# plot_sorted_data("", spikes, intracellular_labels, nr_dim=3, show=True)


