import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_samples, \
    calinski_harabasz_score, davies_bouldin_score, silhouette_score, homogeneity_completeness_v_measure, \
    v_measure_score, fowlkes_mallows_score

from dataset_parsing import simulations_dataset as ds
from metrics import compare_metrics, compute_metrics
from visualization import scatter_plot


def main(program):
    if program == "pca":
        range_min = 10
        range_max = 36
        alignment = 2
        plot_path = f'./figures/pca/'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')
    elif program == "pca_selected":
        range_min = 1
        range_max = 96
        alignment = 2
        min = 30
        max = 50
        plot_path = f'./figures/pca_{min}_{max}/'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue

            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

            spikes = spikes[:, min:max]

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')
    elif program == "pca_variance":
        spikes, labels = ds.get_dataset_simulation(simNr=1)
        pca_2d = PCA(n_components=2)
        pca_features = pca_2d.fit_transform(spikes)
        print("2D:")
        print(pca_2d.explained_variance_ratio_)
        print(pca_2d.explained_variance_)
        pca_2d = PCA(n_components=3)
        pca_features = pca_2d.fit_transform(spikes)
        print("3D:")
        print(pca_2d.explained_variance_ratio_)
        print(pca_2d.explained_variance_)
        pca_2d = PCA(n_components=4)
        pca_features = pca_2d.fit_transform(spikes)
        print("4D:")
        print(pca_2d.explained_variance_ratio_)
        print(pca_2d.explained_variance_)

        pca_2d = PCA(n_components=20)
        pca_features = pca_2d.fit_transform(spikes)

        variance = pca_2d.explained_variance_ratio_  # calculate variance ratios

        var = np.cumsum(np.round(pca_2d.explained_variance_ratio_, decimals=3) * 100)
        print("20D:")
        print(var)

        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(30, 100.5)
        plt.style.context('seaborn-whitegrid')
        plt.plot(var)
        plt.show()
    elif program == "show_spike_shapes":
        range_min = 1
        range_max = 96

        plot_path = f'./figures/spike_shapes/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            folder_path = f'./figures/spike_shapes/sim{simulation_number}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            unique_labels = np.unique(labels)

            for label in unique_labels:
                cluster_spikes = spikes[labels == label]

                plt.figure()
                for i in range(0, len(cluster_spikes)):
                    plt.plot(np.arange(len(cluster_spikes[i])), cluster_spikes[i])
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.title(f"Spikes of cluster {label}")
                plt.savefig(f'./figures/spike_shapes/sim{simulation_number}/' + f"cluster{label}")
    elif program == "benchmark_pca":
        results = []

        range_min = 1
        range_max = 96

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            # spikes = spikes[labels != 0]
            # labels = labels[labels != 0]

            pca_2d = PCA(n_components=2)
            pca_features = pca_2d.fit_transform(spikes)

            pn = 25
            clustering_labels = KMeans(n_clusters=len(np.unique(labels))+1).fit_predict(pca_features)

            hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_labels)

            scores = [adjusted_rand_score(labels, clustering_labels),
                            adjusted_mutual_info_score(labels, clustering_labels),
                            hom,
                            com,
                            vm,
                            calinski_harabasz_score(pca_features, labels),
                            davies_bouldin_score(pca_features, labels),
                            silhouette_score(pca_features, labels)
                            ]

            results.append(scores)

            print(f"{scores[0]:.2f}, {scores[1]:.2f}, "
                  f"{scores[2]:.2f}, {scores[3]:.2f}, "
                  f"{scores[4]:.2f}, {scores[5]:.2f}, "
                  f"{scores[6]:.2f}, {scores[7]:.2f}")


        results = np.array(results)

        mean_values = np.mean(results, axis=0)

        print(f"PCA -> ARI - {mean_values[0]}")
        print(f"PCA -> AMI - {mean_values[1]}")
        print(f"PCA -> Hom - {mean_values[2]}")
        print(f"PCA -> Com - {mean_values[3]}")
        print(f"PCA -> VM - {mean_values[4]}")
        print(f"PCA -> CHS - {mean_values[5]}")
        print(f"PCA -> DBS - {mean_values[6]}")
        print(f"PCA -> SS - {mean_values[7]}")
    elif program == "pca_reconstruction":
        simulation_number = 4
        ### PCA verify
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
        spikes = spikes[labels != 0]
        labels = labels[labels != 0]
        compare_metrics("Full", spikes, labels, len(np.unique(labels)))
        pca_2d = PCA(n_components=20)
        pca_features = pca_2d.fit_transform(spikes)
        print(np.cumsum(np.round(pca_2d.explained_variance_ratio_, decimals=3) * 100))
        compare_metrics("PCA20D", pca_features, labels, len(np.unique(labels)))
        pca_2d = PCA(n_components=2)
        pca_features = pca_2d.fit_transform(spikes)
        compare_metrics("PCA2D", pca_features, labels, len(np.unique(labels)))

        ### PCA spike reconstruction
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        chosen_spike = spikes[0]

        pca = PCA(n_components=2)
        spikes_pca = pca.fit_transform(spikes)
        spikes_projected = pca.inverse_transform(spikes_pca)
        loss = np.sum((spikes - spikes_projected) ** 2, axis=1).mean()

        plt.figure()
        plt.plot(np.arange(len(spikes[0])), spikes[0], label="original")
        # plt.plot(encoded_spike, c="red", marker="o")
        plt.plot(np.arange(len(spikes_projected[0])), spikes_projected[0], label="reconstructed")
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.legend(loc="upper left")
        plt.title(f"Verify PCA 2D")
        plt.show()

# main("pca_variance")
# main("pca")
# main("pca_selected")
# main("show_spike_shapes", sub="")
# main("pca")


def test_scale_normalize():
    spike_verif_path = f'./figures/test_normalized_scaled/spike_verif/'
    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)

    spikes, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=False, scale_spike=False)
    spikes_normalized, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=True, scale_spike=False)
    spikes_scaled, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=False, normalize_spike=False, scale_spike=True)

    plt.plot(np.arange(len(spikes[0])), spikes[0], label="original")
    plt.plot(np.arange(len(spikes_normalized[0])), spikes_normalized[0], label="normalized")
    plt.plot(np.arange(len(spikes_scaled[0])), spikes_scaled[0], label="scaled")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike {0}")
    plt.savefig(f'{spike_verif_path}spikes0')

# test_scale_normalize()


def alignment_show():
    simulation_number = 4

    for alignment in [False, True, 2, 3]:
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=alignment)

        pca_2d = PCA(n_components=2)
        pca_features = pca_2d.fit_transform(spikes)

        scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
        plt.savefig(f'./sim{simulation_number}_pca_align{alignment}')

        plt.figure()
        for i in range(0, len(spikes)):
            plt.plot(np.arange(len(spikes[i])), spikes[i])
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        if alignment == False:
            plt.title(f"Spikes unaligned")
        elif alignment == True:
            plt.title(f"Spikes aligned to average")
        elif alignment == 2:
            plt.title(f"Spikes aligned to global maximum")
        elif alignment == 3:
            plt.title(f"Spikes aligned to global minimum")
        plt.savefig(f'./sim{simulation_number}__spikes_align{alignment}')


def apply_kmeans(simulation_number):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)

    pca_2d = PCA(n_components=2)
    pca_features = pca_2d.fit_transform(spikes)

    scatter_plot.plot('PCA on sim4', pca_features, labels, marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_pca')

    kmeans = KMeans(n_clusters=5, random_state=0).fit(pca_features)

    scatter_plot.plot('K-Means on sim 4', pca_features, kmeans.labels_, marker='o')
    plt.savefig(f'./figures/sim{simulation_number}_kmeans')


# alignment_show()
# apply_kmeans(4)






def evaluate_method(method):
    metrics = []
    for simulation_number in [1,4,16,35]:
        print(simulation_number)
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            met = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            metrics.append(met)
            continue
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=2)
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
    np.savetxt(f"./feature_extraction/autoencoder/analysis/analyze_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")

# evaluate_method('pca')
# evaluate_method('ica')
# evaluate_method('isomap')

