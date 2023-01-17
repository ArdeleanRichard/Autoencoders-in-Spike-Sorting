import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score

from dataset_parsing.read_kampff import read_kampff_c37, read_kampff_c28
from autoencoder import run_autoencoder
from ae_parameters import output_activation, loss_function
from validation.performance import compute_metrics, compute_real_metrics_gt_and_clust_labels
from preprocess.data_scaling import choose_scale
from visualization import scatter_plot


def calculate_WSS(points, kmax):
    # for kmeans cluster estimation
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


def estimate_clusters_on_kampff_data():
    spikes, labels, gt_labels = read_kampff_c37() # 4
    # spikes, labels, gt_labels = read_kampff_c28) # 5
    gt_labels = np.array([gt_labels])

    spikes = np.array(spikes[0])

    pca_2d = PCA(n_components=2)
    data = pca_2d.fit_transform(spikes)


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

# estimate_clusters_on_kampff_data()


def evaluate_kampff_data(data, method):
    metrics = []

    if data == 'Ç28':
        spikes, labels, gt_labels = read_kampff_c28() # 5
        k=4
    elif data == 'C37':
        spikes, labels, gt_labels = read_kampff_c37()  # 4
        k=3
    gt_labels = np.array(gt_labels)

    spikes = np.array(spikes[0])

    if method == 'pca':
        pca_2d = PCA(n_components=2)
        fe_data = pca_2d.fit_transform(spikes)
    if method == 'ica':
        ica_2d = FastICA(n_components=2)
        fe_data = ica_2d.fit_transform(spikes)
    if method == 'isomap':
        iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        fe_data = iso_2d.fit_transform(spikes)

    scatter_plot.plot(f'{method} on {data}', fe_data, gt_labels, marker='o')
    plt.show()
    plt.savefig(f"./figures/analysis/" + f'{method}_{data}')

    met, klabels = compute_real_metrics_gt_and_clust_labels(fe_data, gt_labels, k=k)

    scatter_plot.plot(f'{method}+K-Means on {data}', fe_data, klabels, marker='o')
    plt.savefig(f"./figures/analysis/" + f'{method}_km_{data}')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./figures/analysis/real_C28_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")



# evaluate_kampff_data('C28', 'pca')
evaluate_kampff_data('C37', 'pca')




def calculate_metrics_table():
    method = "PCA"
    method = "ae_normal"
    method = "ae_ortho"
    metrics_saved = []
    for components in [2, 3, 20]:
        for scaling in ["-", "minmax", "divide_amplitude"]:
            # spikes, labels, gt_labels = read_kampff_c28()
            spikes, labels, gt_labels = read_kampff_c37()
            spikes = np.array(spikes)
            labels = np.array(labels)
            if method == "PCA":
                spikes = choose_scale(spikes[0], scaling)
                pca_instance = PCA(n_components=components)
                features = pca_instance.fit_transform(spikes)
            elif method == "ae_normal":
                if components == 20:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="normal",
                                                                  ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                                  code_size=components,
                                                                  output_activation=output_activation,
                                                                  loss_function=loss_function,
                                                                  scale=scaling, shuff=True, nr_epochs=100)

                else:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="normal",
                                                                  ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                                  code_size=components,
                                                                  output_activation=output_activation,
                                                                  loss_function=loss_function,
                                                                  scale=scaling, shuff=True, nr_epochs=100)
            elif method == "ae_ortho":
                if components == 20:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="orthogonal",
                                                                  ae_layers=[60, 50, 40, 30], code_size=components,
                                                                  output_activation=output_activation, loss_function=loss_function,
                                                                  scale=scaling, shuff=True)
                else:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="orthogonal",
                                                                  ae_layers=[40, 20, 10, 5], code_size=components,
                                                                  output_activation=output_activation, loss_function=loss_function,
                                                                  scale=scaling, shuff=True)

            metrics_saved.append(compute_metrics(components, scaling, features, labels, gt_labels))

    np.savetxt(f"./figures/analysis/analysis_c37_{method}.csv", np.around(np.array(metrics_saved), decimals=3).transpose(), delimiter=",")

# calculate_metrics_table()