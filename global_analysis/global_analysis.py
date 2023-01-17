import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from ae_function import run_autoencoder
from dataset_parsing.simulations_dataset import get_dataset_simulation
from validation.performance import compute_metrics_by_kmeans, compare_metrics, compute_metrics, \
    compute_real_metrics_by_kmeans
from visualization import scatter_plot
from dataset_parsing.read_tins_m_data import get_tins_data

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.chdir("../")

def evaluate_simdata(method):
    metrics = []
    for simulation_number in range(1, 96):
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        spikes, labels = get_dataset_simulation(simNr=simulation_number, align_to_peak=2)

        if method == 'pca':
            pca_2d = PCA(n_components=2)
            data = pca_2d.fit_transform(spikes)
        if method == 'ica':
            ica_2d = FastICA(n_components=2, fun='logcosh', max_iter=1000, tol=1e-3)
            data = ica_2d.fit_transform(spikes)
        if method == 'isomap':
            iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
            data = iso_2d.fit_transform(spikes)

        met = compute_metrics_by_kmeans(data, labels, show=False)
        metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./figures/global/{method}.csv", np.array(metrics), fmt="%.3f", delimiter=",")


# evaluate_simdata("pca")
# evaluate_simdata("ica")


# ISOMAP SIMULATIONS
def grid_search_isomap(SIM_NR):
    spikes, labels = get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)
    spikes, gt_labels = shuffle(spikes, labels, random_state=None)

    metrics = []
    for nr_neigh in range(20, 140, 10):
        print(f"{SIM_NR} - {nr_neigh}")
        iso_2d = Isomap(n_neighbors=nr_neigh, neighbors_algorithm='kd_tree', n_components=2, eigen_solver='dense',
                        path_method='FW', metric='minkowski', n_jobs=-1)
        features = iso_2d.fit_transform(spikes)
        met = compute_metrics_by_kmeans(features, gt_labels, show=False)
        metrics.append(met)

    np.savetxt(f"./figures/global/isomap_sim{SIM_NR}_40.csv", np.array(metrics), fmt="%.3f", delimiter=",")


# for simulation_number in range(1, 96):
#     if simulation_number == 25 or simulation_number == 44 or simulation_number == 78 \
#             or simulation_number == 1 or simulation_number == 4 or simulation_number == 16 or simulation_number == 35:
#         continue
#     grid_search_isomap(simulation_number)








# EPOCHS = 50
# LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
# for simulation_number in range(1, 96):
#     if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
#         continue
#
#     for ae_type in ["shallow", "normal", "orthogonal", "ae_pca", "lstm", "fft", "wfft", "tied", "contractive", "ae_pt"]:
#         metrics = []
#
#         features, labels, gt = run_autoencoder(data_type="sim", simulation_number=simulation_number,
#             data=None, labels=None, gt_labels=None, index=None,
#             ae_type="normal", ae_layers=np.array(LAYERS), code_size=2,
#             output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
#             doPlot=False, verbose=0)
#
#         met = compute_metrics_by_kmeans(features, labels, show=False)
#         metrics.append(met)
#
#         np.savetxt(f"./figures/global/{ae_type}_sim{simulation_number}.csv", np.array(metrics), fmt="%.3f", delimiter=",")






