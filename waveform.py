import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.utils import shuffle

from autoencoder import run_autoencoder
from dataset_parsing.read_tins_m_data import get_tins_data
from preprocess.data_scaling import spike_scaling_min_max
from validation.performance import compute_real_metrics_by_kmeans, compute_metrics
from visualization.plot_data import plot_spikes_by_clusters
import visualization.scatter_plot as sp
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def generate_waveforms():
    units_in_channel, labels = get_tins_data()

    # for (i, pn) in list([(4, 25), (6, 40), (17, 15), (26, 30)]):
    for i in list([17]):
        print(i)
        data = units_in_channel[i-1]
        data = np.array(data)
        # data = spike_scaling_min_max(data, min_peak=np.amin(data), max_peak=np.amax(data)) * 2 - 1
        gt_labels = labels[i-1]

        # pca_2d = PCA(n_components=2)
        # X = pca_2d.fit_transform(data)
        # sp.plot(f'K-means on Channel {i}', X, gt_labels, marker='o')
        # print(compute_metrics(X, gt_labels))

        # pca_2d = PCA(n_components=2)
        # X = pca_2d.fit_transform(data)
        # km = KMeans(n_clusters=3).fit(X)
        # sp.plot(f'K-means on Channel {i}', X, km.labels_, marker='o')
        # plt.savefig(f"./figures/waveforms/pca")
        # plot_spikes_by_clusters(data,  km.labels_, title="pca")
        #
        #
        # ica_2d = FastICA(n_components=2)
        # X = ica_2d.fit_transform(data)
        # km = KMeans(n_clusters=3).fit(X)
        # sp.plot(f'K-means on Channel {i}', X, km.labels_, marker='o')
        # plt.savefig(f"./figures/waveforms/ica")
        # plot_spikes_by_clusters(data,  km.labels_, title="ica")
        #
        # iso_2d = Isomap(n_neighbors=20, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        # X = iso_2d.fit_transform(data)
        # km = KMeans(n_clusters=3).fit(X)
        # sp.plot(f'K-means on Channel {i}', X, km.labels_, marker='o')
        # plt.savefig(f"./figures/waveforms/isomap(20)")
        # plot_spikes_by_clusters(data,  km.labels_, title="isomap(20)")
        #

        EPOCHS = 50
        LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
        for i in range(10, 30):
            features, labels, gt_labels = run_autoencoder(data_type="m0", simulation_number=None,
                data=data, labels=gt_labels, gt_labels=gt_labels, index=None,
                ae_type="normal", ae_layers=np.array(LAYERS), code_size=2,
                output_activation='tanh', loss_function='mse', scale="minmax_relu", nr_epochs=EPOCHS, dropout=0.0,
                doPlot=False, verbose=0, shuff=False)

            # features, gt_labels, mue = run_ae_pytorch()

            km = KMeans(n_clusters=3).fit(features)
            sp.plot(f'K-means on Channel {i}', features, km.labels_, marker='o')
            sp.plot(f'K-means on Channel {i}', features, gt_labels, marker='o')
            plt.savefig(f"./figures/waveforms/ae{i}")
            plt.show()
            plot_spikes_by_clusters(data,  km.labels_, title=f"ae{i}")
            print(i, compute_real_metrics_by_kmeans(features, 3)[0], compute_metrics(features, gt_labels))
    # plt.show()


generate_waveforms()
