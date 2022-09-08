import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap

from utils import scatter_plot
from utils.dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
from utils.dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, \
    read_event_codes
from utils.dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_samples, \
    calinski_harabasz_score, davies_bouldin_score, silhouette_score, homogeneity_completeness_v_measure, \
    v_measure_score, fowlkes_mallows_score



def read_kampff_c37():
    DATASET_PATH = './datasets/kampff/c37/Spikes/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1]
    #*# unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

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

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels


def read_kampff_c28():
    DATASET_PATH = './datasets/kampff/c28/units/'
    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

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

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

def real_feature_extraction():
    spikes, labels, gt_labels = read_kampff_c37() # 4
    # spikes, labels, gt_labels = read_kampff_c28) # 5
    gt_labels = np.array([gt_labels])

    spikes = np.array(spikes[0])

    pca_2d = PCA(n_components=2)
    data = pca_2d.fit_transform(spikes)

    from sklearn.metrics import silhouette_score

    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))

    print(sil)
    test = calculate_WSS(data, 10)
    print(test)

# real_feature_extraction()

def compute_real_metrics(features, gt_labels, k):
    kmeans_labels1 = KMeans(n_clusters=k).fit_predict(features)
    kmeans_labels1 = np.array(kmeans_labels1)
    gt_labels = np.array(gt_labels)

    metrics = []
    metrics.append(davies_bouldin_score(features, kmeans_labels1))
    metrics.append(calinski_harabasz_score(features, kmeans_labels1))
    metrics.append(silhouette_score(features, kmeans_labels1))
    metrics.append(davies_bouldin_score(features, gt_labels))
    metrics.append(calinski_harabasz_score(features, gt_labels))
    metrics.append(silhouette_score(features, gt_labels))

    return metrics, kmeans_labels1


def evaluate_method1(method):
    metrics = []

    spikes, labels, gt_labels = read_kampff_c37()  # 4
    # spikes, labels, gt_labels = read_kampff_c28) # 5
    gt_labels = np.array(gt_labels)

    spikes = np.array(spikes[0])

    if method == 'pca':
        pca_2d = PCA(n_components=2)
        data = pca_2d.fit_transform(spikes)
    if method == 'ica':
        ica_2d = FastICA(n_components=2)
        data = ica_2d.fit_transform(spikes)
    if method == 'isomap':
        iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        data = iso_2d.fit_transform(spikes)

    scatter_plot.plot(f'{method} on C37', data, gt_labels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{method}_c37')

    met, klabels = compute_real_metrics(data, gt_labels, 3)

    scatter_plot.plot(f'K-Means on C37', data, klabels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{method}_km_c37')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./feature_extraction/autoencoder/analysis/real_c37_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")


def evaluate_method2(method):
    metrics = []

    spikes, labels, gt_labels = read_kampff_c28() # 5
    gt_labels = np.array(gt_labels)

    spikes = np.array(spikes[0])

    if method == 'pca':
        pca_2d = PCA(n_components=2)
        data = pca_2d.fit_transform(spikes)
    if method == 'ica':
        ica_2d = FastICA(n_components=2)
        data = ica_2d.fit_transform(spikes)
    if method == 'isomap':
        iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        data = iso_2d.fit_transform(spikes)

    scatter_plot.plot(f'{method} on C28', data, gt_labels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{method}_C28')

    met, klabels = compute_real_metrics(data, gt_labels, k=4)

    scatter_plot.plot(f'K-Means on C28', data, klabels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{method}_km_C28')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./feature_extraction/autoencoder/analysis/real_C28_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")


def get_M045_009():
    DATASET_PATH = './datasets/M045_0009/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 58

    timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)

    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)

    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=33)

    return units_in_channels, labels


def compute_real_metrics2(features, k):
    kmeans_labels1 = KMeans(n_clusters=k).fit_predict(features)
    kmeans_labels1 = np.array(kmeans_labels1)

    metrics = []
    metrics.append(davies_bouldin_score(features, kmeans_labels1))
    metrics.append(calinski_harabasz_score(features, kmeans_labels1))
    metrics.append(silhouette_score(features, kmeans_labels1))

    return metrics, kmeans_labels1


def evaluate(method):
    metrics = []
    index = 6
    units_in_channel, labels = get_M045_009()
    spikes = units_in_channel[index - 1]
    spikes = np.array(spikes)

    if method == 'pca':
        pca_2d = PCA(n_components=2)
        data = pca_2d.fit_transform(spikes)
    if method == 'ica':
        ica_2d = FastICA(n_components=2)
        data = ica_2d.fit_transform(spikes)
    if method == 'isomap':
        iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        data = iso_2d.fit_transform(spikes)


    met, klabels = compute_real_metrics2(data, k=4)

    scatter_plot.plot(f'K-Means on C28', data, klabels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'real_m045_{index}_{method}_km')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./feature_extraction/autoencoder/analysis/real_m045_{index}_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")


evaluate('pca')
evaluate('ica')
evaluate('isomap')