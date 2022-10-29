from constants import DATA_FOLDER_PATH
from dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel



def get_tins_data(name="M045_0009"):
    DATASET_PATH = DATA_FOLDER_PATH + f'/TINS/{name}/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 58

    timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)

    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)

    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=33)

    return units_in_channels, labels