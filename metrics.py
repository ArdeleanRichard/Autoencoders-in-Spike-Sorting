import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score, homogeneity_completeness_v_measure

from ae_variants import calculate_metrics_specific
from dataset_parsing.read_kampff import read_kampff_c28, read_kampff_c37
from preprocess.data_scaling import choose_scale
from validation.scores import purity_score
from visualization import scatter_plot


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


def compute_real_metrics(features, k):
    try:
        kmeans_labels1 = KMeans(n_clusters=k).fit_predict(features)
        kmeans_labels1 = np.array(kmeans_labels1)

        metrics = []
        metrics.append(davies_bouldin_score(features, kmeans_labels1))
        metrics.append(calinski_harabasz_score(features, kmeans_labels1))
        metrics.append(silhouette_score(features, kmeans_labels1))
    except ValueError:
        metrics = [0,0,0]
        kmeans_labels1 = np.zeros((len(features),))

    return metrics, kmeans_labels1



def compute_real_metrics_gt_and_clust_labels(features, gt_labels, k):
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


def feature_scores(labels, clustering_labels):
    hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_labels)

    scores = [adjusted_rand_score(labels, clustering_labels),
              adjusted_mutual_info_score(labels, clustering_labels),
              hom,
              com,
              vm,
              calinski_harabasz_score(clustering_labels, labels),
              davies_bouldin_score(clustering_labels, labels),
              silhouette_score(clustering_labels, labels)
              ]

    return scores


def compare_features(autoencoder_features, pca_features, plot_path, simulation_number, labels):
    unique_labels = np.unique(labels)
    clustering_ae_labels = KMeans(n_clusters=len(unique_labels)).fit_predict(autoencoder_features)
    clustering_pca_labels = KMeans(n_clusters=len(unique_labels)).fit_predict(pca_features)

    scatter_plot.plot('Kmeans' + str(len(autoencoder_features)), autoencoder_features, clustering_ae_labels,
                      marker='o')
    plt.savefig(plot_path + f'gt_model_sim{simulation_number}_ae_kmeans')

    scatter_plot.plot('kmeans' + str(len(pca_features)), pca_features, clustering_pca_labels,
                      marker='o')
    plt.savefig(plot_path + f'gt_model_sim{simulation_number}_pca_kmeans')

    scores_ae = feature_scores(labels, clustering_ae_labels)

    print(f"{simulation_number}, {len(unique_labels)}, "
          f"{scores_ae[0]:.2f}, {scores_ae[1]:.2f}, "
          f"{scores_ae[2]:.2f}, {scores_ae[3]:.2f}, "
          f"{scores_ae[4]:.2f}, {scores_ae[5]:.2f}, "
          f"{scores_ae[6]:.2f}, {scores_ae[7]:.2f}")

    scores_pca = feature_scores(labels, clustering_pca_labels)

    print(f"{simulation_number}, {len(unique_labels)}, "
          f"{scores_pca[0]:.2f}, {scores_pca[1]:.2f}, "
          f"{scores_pca[2]:.2f}, {scores_pca[3]:.2f}, "
          f"{scores_pca[4]:.2f}, {scores_pca[5]:.2f}, "
          f"{scores_pca[6]:.2f}, {scores_pca[7]:.2f}")

    print(f"{simulation_number}, {len(unique_labels)}, "
          f"{scores_ae[0] - scores_pca[0]:.2f}, {scores_ae[1] - scores_pca[1]:.2f}, "
          f"{scores_ae[2] - scores_pca[2]:.2f}, {scores_ae[3] - scores_pca[3]:.2f}, "
          f"{scores_ae[4] - scores_pca[4]:.2f}, {scores_ae[5] - scores_pca[5]:.2f}, "
          f"{scores_ae[6] - scores_pca[6]:.2f}, {scores_ae[7] - scores_pca[7]:.2f}")

    return scores_ae, scores_pca


def plot_metrics_clustering_eval(title, pca, oae, xlabel, ylabel):
    max_saved = max(oae[5], pca[5])
    oae[5] = oae[5] / max_saved
    pca[5] = pca[5] / max_saved

    max_saved = max(oae[4], pca[4])
    oae[4] = oae[4] / max_saved
    pca[4] = pca[4] / max_saved

    plt.title(title)
    plt.plot([0, 1], [0, 1], 'g--')
    plt.scatter(pca, oae, c=[0,0,0,0,1,0,0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_metrics_fe_eval(title, pca, oae, xlabel, ylabel):
    max_saved = max(oae[0], pca[0])
    oae[0] = oae[0] / max_saved
    pca[0] = pca[0] / max_saved

    max_saved = max(oae[1], pca[1])
    oae[1] = oae[1] / max_saved
    pca[1] = pca[1] / max_saved

    plt.title(title)
    plt.plot([-0.5, 1], [-0.5, 1], 'g--')
    plt.scatter(pca, oae, c=[1,0,0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()





def create_plot_metrics(data="C37", components=2, scaling="minmax"):
    if data=="C28":
        spikes, labels, gt_labels = read_kampff_c28()
    elif data=="C37":
        spikes, labels, gt_labels = read_kampff_c37()


    # for components in [2, 3, 20]:
    # for scaling in ["-", "minmax", "divide_amplitude"]:

    method1 = "PCA"
    # method2 = "AE"
    method2 = "AE"
    non_determenistic_runs = 10
    metrics1 = calculate_metrics_specific(spikes, labels, gt_labels, method=method1, components=components, scaling=scaling)

    sum = []
    for i in range(non_determenistic_runs):
        metrics2 = calculate_metrics_specific(spikes, labels, gt_labels, method=method2, components=components, scaling=scaling)
        sum.append(metrics2)

    sum = np.array(sum)
    metrics2 = np.mean(sum, axis=0)

    plot_metrics_clustering_eval(f"{data} - {components}D - {scaling}", metrics1[:7], metrics2[:7], method1, method2)
    plot_metrics_fe_eval(f"{data} - {components}D - {scaling}", metrics1[7:], metrics2[7:], method1, method2)

