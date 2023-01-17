import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import Isomap

from ae_function import run_autoencoder
from dataset_parsing.read_tins_m_data import get_tins_data
from visualization import scatter_plot
from dataset_parsing import simulations_dataset as ds

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def run_methods_on_synthetic_data():
    SIM_NR = 4
    EPOCHS = 100
    LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]

    data, labels = ds.get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)

    pca_2d = PCA(n_components=2)
    features = pca_2d.fit_transform(data)
    scatter_plot.plot(f'PCA', features, labels, marker='o')
    plt.show()

    ica_2d = FastICA(n_components=2, fun='logcosh', max_iter=500, tol=1e-3)
    features = ica_2d.fit_transform(data)
    scatter_plot.plot(f'ICA', features, labels, marker='o')
    plt.show()

    iso_2d = Isomap(n_neighbors=90, neighbors_algorithm='kd_tree', n_components=2, eigen_solver='dense', path_method='FW', metric='minkowski', n_jobs=-1)
    features = iso_2d.fit_transform(data)
    scatter_plot.plot(f'Isomap', features, labels, marker='o')
    plt.show()

    for ae_type in ["shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
        run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                        data=None, labels=None, gt_labels=None, index=None,
                        ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
                        output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                        doPlot=True, verbose=0)




def run_methods_on_real_data():
    # # RUN REAL
    EPOCHS = 100
    LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
    CHANNEL = 17

    units_in_channel, labels = get_tins_data()
    data = units_in_channel[CHANNEL - 1]
    data = np.array(data)
    labels = np.array(labels[CHANNEL-1])

    pca_2d = PCA(n_components=2)
    features = pca_2d.fit_transform(data)
    scatter_plot.plot(f'PCA', features, labels, marker='o')
    plt.show()

    ica_2d = FastICA(n_components=2, fun='logcosh', max_iter=500, tol=1e-3)
    features = ica_2d.fit_transform(data)
    scatter_plot.plot(f'ICA', features, labels, marker='o')
    plt.show()

    iso_2d = Isomap(n_neighbors=20, neighbors_algorithm='kd_tree', n_components=2, eigen_solver='dense', path_method='FW', metric='minkowski', n_jobs=-1)
    features = iso_2d.fit_transform(data)
    scatter_plot.plot(f'Isomap', features, labels, marker='o')
    plt.show()

    for ae_type in ["shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
        features, _, gt = run_autoencoder(data_type="m0", simulation_number=None,
                        data=data, labels=None, gt_labels=labels, index=None,
                        ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
                        output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                        doPlot=False, verbose=0, shuff=True)

        scatter_plot.plot(f'AE {ae_type}', features, gt, marker='o')
        plt.show()






if __name__ == '__main__':
    run_methods_on_synthetic_data()
    # run_methods_on_real_data()











