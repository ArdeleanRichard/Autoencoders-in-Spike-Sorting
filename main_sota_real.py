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
# evaluate('isomap')