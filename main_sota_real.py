import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap

from visualization import scatter_plot
from dataset_parsing.read_tins_m_data import get_tins_data
from metrics import compute_real_metrics


def evaluate(method):
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
        iso_2d = Isomap(n_neighbors=100, n_components=2, eigen_solver='arpack', path_method='D', n_jobs=-1)
        data = iso_2d.fit_transform(spikes)


    met, klabels = compute_real_metrics(data, k=4)

    scatter_plot.plot(f'K-Means on C28', data, klabels, marker='o')
    plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'real_m045_{index}_{method}_km')

    metrics.append(met)

    metrics = np.array(metrics)
    np.savetxt(f"./feature_extraction/autoencoder/analysis/real_m045_{index}_{method}.csv",
               np.around(np.array(metrics), decimals=3).transpose(), delimiter=",")


evaluate('pca')
evaluate('ica')
# evaluate('isomap')