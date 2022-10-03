import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils import shuffle
import os

from feature_extraction.autoencoder.scaling import spike_scaling_min_max, spike_scaling_ignore_amplitude, get_spike_energy



def get_type(on_type, fft_real, fft_imag):
    if on_type == "real":
        spikes = fft_real
    elif on_type == "imag":
        spikes = fft_imag
    elif on_type == "magnitude":
        spikes = np.sqrt(fft_real * fft_real + fft_imag * fft_imag)
    elif on_type == "power":
        spikes = fft_real * fft_real + fft_imag * fft_imag
    elif on_type == "phase":
        spikes = np.arctan2(fft_imag, fft_real)
    elif on_type == "concatenated":
        power = fft_real * fft_real + fft_imag * fft_imag
        phase = np.arctan2(fft_imag, fft_real)
        spikes = np.concatenate((power, phase), axis=1)

    return spikes

def choose_scale(spikes, scale_type):
    if scale_type == 'minmax':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    if scale_type == 'minmax_relu':
        spikes = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
    if scale_type == 'minmax_spp':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 4) - 3
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == '-1+1':
        spikes = spike_scaling_min_max(spikes, min_peak=-1, max_peak=1)
        # plot_spikes(spikes_scaled, title='scale_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'scaler':
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        # plot_spikes(spikes_scaled, title='scale_sklearn_0_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude':
        # SCALE IGNORE AMPLITUDE
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_no_amplitude', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude_add_amplitude':
        # SCALE IGNORE AMPLITUDE ADDED FEATURE AMPLITUDE
        amplitudes = np.amax(spikes, axis=1)
        amplitudes = amplitudes.reshape((-1,1))
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        spikes = np.hstack((spikes, amplitudes))
    elif scale_type == 'add_energy':
        # SCALED ADDED FEATURE ENERGY
        spikes_energy = get_spike_energy(spikes)
        spikes_energy = spikes_energy.reshape((-1,1))
        spikes = np.hstack((spikes, spikes_energy))
        # print(spikes.shape)
    elif scale_type == 'divide_amplitude':
        amplitudes = np.amax(spikes, axis=1)
        print(len(amplitudes))
        amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
        spikes = spikes / amplitudes
    elif scale_type == 'scale_no_energy_loss':
        # print(spikes)
        for spike in spikes:
            spike[spike < 0] = spike[spike < 0] / abs(np.amin(spike))
            spike[spike > 0] = spike[spike > 0] / np.amax(spike)

    return spikes

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)



def compare_metrics(Data, X, y, n_clusters):
    # dataset

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    #metric - ARI
    print(f"{Data} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - SS: "
          f"KMeans={silhouette_score(X, kmeans.labels_):.3f}\t")

    print(f"{Data} - CHS: "
          f"KMeans={calinski_harabasz_score(X, kmeans.labels_):.3f}\t")

    print(f"{Data} - DBS: "
          f"KMeans={davies_bouldin_score(X, kmeans.labels_):.3f}\t")

    print()



##### CHECK CLUSTERING METRICS AND FE METRICS
def compute_metrics(features, gt_labels):
    try:
        kmeans_labels1 = KMeans(n_clusters=len(np.unique(gt_labels))).fit_predict(features)
        kmeans_labels1 = np.array(kmeans_labels1)
        gt_labels = np.array(gt_labels)

        metrics = []
        metrics.append(adjusted_rand_score(kmeans_labels1, gt_labels))
        metrics.append(adjusted_mutual_info_score(kmeans_labels1, gt_labels))
        metrics.append(v_measure_score(kmeans_labels1, gt_labels))
        metrics.append(fowlkes_mallows_score(kmeans_labels1, gt_labels))
        metrics.append(davies_bouldin_score(features, kmeans_labels1))
        metrics.append(calinski_harabasz_score(features, kmeans_labels1))
        metrics.append(silhouette_score(features, kmeans_labels1))
        metrics.append(davies_bouldin_score(features, gt_labels))
        metrics.append(calinski_harabasz_score(features, gt_labels))
        metrics.append(silhouette_score(features, gt_labels))
    except ValueError:
        metrics = [0,0,0,0,0,0,0,0,0,0]

    return metrics