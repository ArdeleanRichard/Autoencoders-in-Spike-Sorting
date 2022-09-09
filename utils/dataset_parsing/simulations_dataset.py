import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from peakdetect import peakdetect
import matplotlib.pyplot as plt
from utils import scatter_plot

def get_dataset_simulation_features(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset with 2 chosen feaWWtures (amplitude and distance between min peaks)
    The amplitude can be used to find noise (noise has a lower amplitude than the other clusters, depending on similuation from 0.75->1)
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_features: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    spikes_features = np.empty((len(spikes), 2))

    for i in range(len(spikes)):
        # print(i)
        max_peaks, min_peaks = peakdetect(spikes[i], range(spike_length), lookahead=1)
        # print(max_peaks)
        # print(min_peaks)
        max_peaks = np.array(max_peaks)

        amplitude_information = np.argmax(max_peaks[:, 1])
        amplitude_position = max_peaks[amplitude_information][0]
        spike_amplitude = max_peaks[amplitude_information][1]

        spike_distance = 0

        if amplitude_position < min_peaks[0][0]:
            spike_distance = min_peaks[0][0] - 0
        else:
            for j in range(0, len(min_peaks)):
                if j + 1 >= len(min_peaks):
                    spike_distance = 79 - min_peaks[j][0]
                    # plt.figure()
                    # plt.plot(spikes[i])
                    # plt.savefig(f"./figures/FirstSpike{i}")
                    break
                else:
                    if min_peaks[j][0] < amplitude_position < min_peaks[j + 1][0]:
                        spike_distance = min_peaks[j + 1][0] - min_peaks[j][0]
                        break

        spikes_features[i] = [spike_amplitude, spike_distance]

        # if spike_amplitude < 0.5:
        #     plt.figure()
        #     plt.plot(spikes[i])
        #     plt.savefig(f"./figures/Noise{i},{spike_distance}")

    return spikes_features, labels


def get_dataset_simulation_pca_2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_2d: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    # apply pca_nonoise
    pca_2d = PCA(n_components=2)
    spikes_pca_2d = pca_2d.fit_transform(spikes)
    # getDatasetSimulationPlots(spikes, spikes_pca_2d, spikes_pca_3d, labels)

    # np.save('79_ground_truth', label)
    # np.save('79_x', spikes_reduced[:, 0])
    # np.save('79_y', spikes_reduced[:, 1])

    return spikes_pca_2d, labels


# spike extraction options
# original sampling rate 96KHz, with each waveform at 316 points(dimensions/features)
# downsampled to 24KHz, (regula-3-simpla) => 79 points (de aici vine 79 de mai jos)
def get_dataset_simulation_pca_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 3 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_3d: matrix - the 3-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    # apply pca_nonoise
    pca_3d = PCA(n_components=3)
    spikes_pca_3d = pca_3d.fit_transform(spikes)

    return spikes_pca_3d, labels


def get_dataset_simulation(simNr, spike_length=79, align_to_peak=True, normalize_spike=False, scale_spike=False):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary = loadmat('./datasets/simulations/simulation_' + str(simNr) + '.mat')
    ground_truth_dictionary = loadmat('./datasets/simulations/ground_truth.mat')

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    # each spike will contain the first 79 points from the data after it has started
    spikes = spike_preprocess(data, start, spike_length, align_to_peak, normalize_spike, scale_spike)

    return spikes, labels


def get_signal_simulation(simNr):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary = loadmat('./datasets/simulations/simulation_' + str(simNr) + '.mat')
    ground_truth_dictionary = loadmat('./datasets/simulations/ground_truth.mat')

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    return data, start, labels


def spike_preprocess(signal, spike_start, spike_length, align_to_peak, normalize_spikes, scale_spikes, index_to_align_peak=39):
    spikes = spike_extract(signal, spike_start, spike_length)

    if scale_spikes == True:
        spikes_std = np.zeros_like(spikes)
        min_peak = np.amin(spikes)
        max_peak = np.amax(spikes)
        for col in range(len(spikes[0])):
            spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)
        spikes = spikes_std

    # align to max
    if isinstance(align_to_peak, bool) and align_to_peak:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmax(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        avg_peak = np.floor(np.mean(peak_ind))
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (avg_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)
    # align_to_peak == 2 means that it will align the maximum point (Na+ polarization) to index 39 (the middle)
    elif align_to_peak == 2:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmax(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        # index_to_align_peak = 39
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (index_to_align_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)
    # align_to_peak == 3 means that it will align the minimum point (K+ polarization) to index 39 (the middle)
    elif align_to_peak == 3:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmin(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        # index_to_align_peak = 39
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (index_to_align_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)

    # normalize spikes using Z-score: (value - mean)/ standard deviation
    if normalize_spikes:
        normalized_spikes = [(spike - np.mean(spike)) / np.std(spike) for spike in spikes]
        normalized_spikes = np.array(normalized_spikes)
        return normalized_spikes
    return spikes


def get_dataset_simulation_noise(simNr, spike_length=79, noise_length=79):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary = loadmat('./datasets/simulations/simulation_' + str(simNr) + '.mat')
    ground_truth_dictionary = loadmat('./datasets/simulations/ground_truth.mat')

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    # each spike will contain the first 79 points from the data after it has started
    noise = noise_extract(data, start, spike_length, noise_length)

    return noise


def noise_extract(signal, spike_start, spike_length, noise_length):
    """
    Extract the noise from the signal knowing where the spikes start and their length
    :param signal: matrix - height for each point of the spikes
    :param spike_start: vector - each entry represents the first point of a spike
    :param spike_length: integer - constant, 79

    :returns spikes: matrix - each row contains 79 points of one spike
    """
    signal = np.array(signal)

    mask = np.ones((len(signal),))
    for i in range(len(spike_start)):
        mask[spike_start[i] - int(spike_length *1/2): spike_start[i] + int(spike_length *3/2)] = 0

    signal = signal[mask.astype(np.bool)]

    noise = np.reshape(signal[:-(len(signal) % spike_length)], (-1, spike_length))

    return noise


def spike_extract(signal, spike_start, spike_length):
    """
    Extract the spikes from the signal knowing where the spikes start and their length
    :param signal: matrix - height for each point of the spikes
    :param spike_start: vector - each entry represents the first point of a spike
    :param spike_length: integer - constant, 79

    :returns spikes: matrix - each row contains 79 points of one spike
    """
    spikes = np.zeros([len(spike_start), spike_length])

    for i in range(len(spike_start)):
        spikes[i, :] = signal[spike_start[i]: spike_start[i] + spike_length]

    return spikes


"""
----------------------------------------------------------
OBSOLETE
----------------------------------------------------------
"""

# FOR SIMULATION 79
# dataset.mat in key 'data' == simulation_97.mat in key 'data'
# dataset.mat in key 'ground_truth' == ground_truth.mat in key 'spike_classes'[78]
# dataset.mat in key 'start_spikes' == ground_truth.mat in key 'spike_first_sample'[78]
# dataset.mat in key 'spike_wf' == ground_truth.mat in key 'su_waveforms'[78] (higher precision in GT)
def getDatasetSim79():
    """
    Load the dataset Simulation79
    :param None

    :returns spikes_pca_2d: matrix - the points that have been taken through 2D PCA
    :returns labels: vector - the vector of labels for simulation79
    """
    dictionary = loadmat('./datasets/simulations/dataset.mat')

    # dataset file is a dictionary (the data has been extracted from ground_truth.mat and simulation_79.mat), containing following keys:
    # ground_truth (shape = 14536): the labels of the points
    # start_spikes (shape = 14536): the start timestamp of each spike
    # data (shape = 14400000): the raw spikes with all their points
    # spike_wf (shape = (20, 316)): contains the form of each spike (20 spikes, each with 316 dimensions/features) NOT USED YET

    labels = dictionary['ground_truth'][0, :]
    start = dictionary['start_spikes'][0, :]
    data = dictionary['data'][0, :]

    # spike extraction options
    # original sampling rate 96KHz, with each waveform at 316 points(dimensions/features)
    # downsampled to 24KHz, (regula-3-simpla) => 79 points (de aici vine 79 de mai jos)
    spike_length = 79  # length of spikes in number of samples
    align_to_peak = False  # aligns each spike to it's maximum value
    normalize_spike = False  # applies z-scoring normalization to each spike

    # each spike will contain the first 79 points from the data after it has started
    spikes = spike_preprocess(data, start, spike_length, align_to_peak, normalize_spike, labels)

    # apply pca_nonoise
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    spikes_pca_2d = pca_2d.fit_transform(spikes)
    spikes_pca_3d = pca_3d.fit_transform(spikes)

    return spikes_pca_2d, labels


def getDatasetSim97Plots(spikes, spike_pca_2d, spikes_pca_3d, labels):
    # plot some spikes
    ind = np.random.randint(0, len(labels), [20])
    plt.plot(np.transpose(spikes[ind, :]))
    plt.show()

    # plot all spikes from one unit
    unit = 15
    ind = np.squeeze(np.argwhere(labels == unit))
    plt.plot(np.transpose(spikes[ind, :]))
    plt.title('Unit {}'.format(unit))
    plt.show()

    plotSimulation_PCA2D_grid(spike_pca_2d, labels)

    plotSimulation_PCA3D(spikes_pca_3d, labels)


def plotSimulation_PCA2D(spike_pca_2d, labels):
    # plot scatter of pca_nonoise
    plt.scatter(spike_pca_2d[:, 0], spike_pca_2d[:, 1], c=labels, marker='x', cmap='brg')
    plt.show()


def plotSimulation_PCA2D_grid(spike_pca_2d, labels):
    # plot scatter of pca_nonoise
    scatter_plot.plot_grid('Sim79Gridded', spike_pca_2d, labels + 1, 25, marker='x')
    plt.show()


def plotSimulation_PCA3D(spikes_pca_3d, labels):
    # plot scatter of pca_nonoise in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(spikes_pca_3d[:, 0], spikes_pca_3d[:, 1], spikes_pca_3d[:, 2], c=labels, marker='x', cmap='brg')
    plt.show()





