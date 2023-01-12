import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from dataset_parsing.simulations_dataset import get_dataset_simulation
from validation.performance import compute_metrics_by_kmeans, compare_metrics, compute_metrics, compute_real_metrics_by_kmeans
from visualization import scatter_plot
from dataset_parsing.read_tins_m_data import get_tins_data


def evaluate_realdata(method):
    metrics = []
    index = 6
    units_in_channel, labels = get_tins_data()
    spikes = units_in_channel[index - 1]
    spikes = np.array(spikes)

    if method == 'pca':
        pca_2d = PCA(n_components=2)
        data = pca_2d.fit_transform(spikes)
    if method == 'ica':
        ica_2d = FastICA(n_components=2)
        data = ica_2d.fit_transform(spikes)
    if method == 'isomap':
        iso_2d = Isomap(n_neighbors=20, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        data = iso_2d.fit_transform(spikes)


    met, klabels = compute_real_metrics_by_kmeans(data, k=4)

    scatter_plot.plot(f'K-Means on C28', data, klabels, marker='o')
    plt.savefig(f"./figures/analysis/" + f'real_m045_{index}_{method}_km')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./figures/analysis/real_m045_{index}_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")


# evaluate_realdata('pca')
# evaluate_realdata('ica')
# evaluate_realdata('isomap')



def evaluate_simdata(method):
    metrics = []
    for simulation_number in [1,4,16,35]:
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            met = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            metrics.append(met)
            continue
        spikes, labels = get_dataset_simulation(simNr=simulation_number, align_to_peak=2)
        if method == 'pca':
            pca_2d = PCA(n_components=2)
            data = pca_2d.fit_transform(spikes)
        if method == 'ica':
            ica_2d = FastICA(n_components=2)
            data = ica_2d.fit_transform(spikes)
        if method == 'isomap':
            iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
            data = iso_2d.fit_transform(spikes)

        scatter_plot.plot(f'{method} on Sim{simulation_number}', data, labels, marker='o')
        plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{method}_{simulation_number}')

        met = compute_metrics(data, labels)
        metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./feature_extraction/autoencoder/analysis/analyze_{method}.csv", np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")

# evaluate_simdata('pca')
# evaluate_simdata('ica')
# evaluate_simdata('isomap')


def grid_search_ica(SIM_NR):
    metrics = []
    spikes, labels = get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)
    spikes, gt_labels = shuffle(spikes, labels, random_state=None)

    for fun in ['logcosh', 'exp', 'cube']:
        for max_iter in [200, 300, 400, 500]:
            for tol in [1e-3, 1e-4, 1e-5]:
                print(fun, max_iter, tol)
                ica_2d = FastICA(n_components=2, fun=fun, max_iter=max_iter, tol=tol)
                features = ica_2d.fit_transform(spikes)
                met = compute_metrics_by_kmeans(features, gt_labels, show=False)
                metrics.append(met)

    np.savetxt(f"./validation/sim{SIM_NR}_ica_grid_search.csv", np.array(metrics), fmt="%.3f", delimiter=",")



def grid_search_isomap(SIM_NR):
    spikes, labels = get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)
    spikes, gt_labels = shuffle(spikes, labels, random_state=None)


    metrics = []
    for nr_neigh in range(5, 200, 5):
        # metrics = []
        iso_2d = Isomap(n_neighbors=nr_neigh, neighbors_algorithm='kd_tree', n_components=2, eigen_solver='dense',
                        path_method='FW', metric='minkowski', n_jobs=-1)
        features = iso_2d.fit_transform(spikes)
        met = compute_metrics_by_kmeans(features, gt_labels, show=False)
        metrics.append(met)

        # for neighb_alg in ['auto', 'brute', 'kd_tree', 'ball_tree']:
        #     for eigen_solver in ['auto', 'arpack', 'dense']:
        #         for path_method in ['auto', 'FW', 'D']:
        #             for metric in ['minkowski', 'euclidean', 'cosine', 'cityblock']:
        #                 if eigen_solver == 'dense':
        #                     print(nr_neigh, eigen_solver, path_method, metric)
        #                     iso_2d = Isomap(n_neighbors=nr_neigh, neighbors_algorithm=neighb_alg, n_components=2, eigen_solver=eigen_solver, path_method=path_method, metric=metric, n_jobs=-1)
        #                     features = iso_2d.fit_transform(spikes)
        #                     met = compute_metrics_by_kmeans(features, gt_labels, show=False)
        #                     metrics.append(met)
        #                 else:
        #                     for max_iter in [200, 300, 400, 500]:
        #                         for tol in [1e-3, 1e-4, 1e-5]:
        #                             print(nr_neigh, eigen_solver, path_method, metric, max_iter, tol)
        #                             iso_2d = Isomap(n_neighbors=nr_neigh, neighbors_algorithm=neighb_alg, n_components=2, eigen_solver=eigen_solver, path_method=path_method, metric=metric, n_jobs=-1)
        #                             features = iso_2d.fit_transform(spikes)
        #                             met = compute_metrics_by_kmeans(features, gt_labels, show=False)
        #                             metrics.append(met)

        # np.savetxt(f"./validation/ica_grid_search_n{nr_neigh}_nalg.csv", np.array(metrics), fmt="%.3f", delimiter=",")
    np.savetxt(f"./validation/sim{SIM_NR}_isomap_grid_search.csv", np.array(metrics), fmt="%.3f", delimiter=",")


# grid_search_ica(1)
# grid_search_ica(4)
# grid_search_ica(16)
# grid_search_ica(35)


# grid_search_isomap(1)
# grid_search_isomap(4)
# grid_search_isomap(16)
# grid_search_isomap(35)
