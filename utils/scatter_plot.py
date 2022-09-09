import math

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from sklearn import preprocessing
from sklearn.decomposition import PCA

from utils import constants as cs
from utils.dataset_parsing import simulations_dataset as ds
import seaborn as sns


def plot(title, X, labels=None, plot=True, marker='o'):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    if plot:
        nrDim = len(X[0])
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            if labels is None:
                plt.scatter(X[:, 0], X[:, 1], marker=marker, edgecolors='k')
            else:
                try:
                    label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
                except KeyError:
                    print('Too many labels! Using default colors...\n')
                    label_color = [l for l in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k')
        if nrDim == 3:
            ax = fig.add_subplot(projection='3d')
            if labels is None:
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, edgecolors='k')
            else:
                try:
                    label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
                except KeyError:
                    print('Too many labels! Using default colors...\n')
                    label_color = [l for l in labels]
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_color, marker=marker, edgecolors='k')

def data_preprocessing(spikes, pn, adaptivePN=False):
    if adaptivePN == True:
        # feature_variance = np.var(spikes, axis=0)
        # print(feature_variance)

        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        feature_variance = np.var(spikes, axis=0)
        # print(feature_variance)

        # pca = PCA(n_components=2)
        # pca.fit(spikes)
        # feature_variance = pca.explained_variance_ratio_
        feature_variance = feature_variance / np.amax(feature_variance)
        feature_variance = feature_variance * pn
        # feature_variance[1] = feature_variance[1] * 3
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        spikes = spikes * np.array(feature_variance)
        # print(feature_variance)

        return spikes, feature_variance

    spikes = preprocessing.MinMaxScaler((0, pn)).fit_transform(spikes)

    return spikes, pn

def plot_grid(title, X, pn, labels=None, plot=True, marker='o', adaptivePN=False):
    """
    Plots the dataset with grid
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    X, pn = data_preprocessing(X, pn, adaptivePN=adaptivePN)
    if plot:
        nrDim = len(X[0])
        label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            ax = fig.gca()
            if not isinstance(pn, int):
                ax.set_xticks(np.arange(0, pn[0], 1))
                ax.set_yticks(np.arange(0, pn[1], 1))
            else:
                ax.set_xticks(np.arange(0, pn, 1))
                ax.set_yticks(np.arange(0, pn, 1))
            plt.scatter(X[:, 0], X[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.set_xticks(np.arange(0, pn, 1))
            # ax.set_zticks(np.arange(0, pn, 1))
            # ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, s=25)
            # plt.grid(True)


def plot_clusters(spikes, labels=None, title="", save_folder=""):
    if spikes.shape[1] == 2:
        plot(title, spikes, labels)
        if save_folder != "":
            plt.savefig('./figures/' + save_folder + "/" + title)
        plt.show()
    elif spikes.shape[1] == 3:
        fig = px.scatter_3d(spikes, x=spikes[:, 0], y=spikes[:, 1], z=spikes[:, 2], color=labels.astype(str))
        fig.update_layout(title=title)
        fig.show()


def plot_centers(title, X, clusterCenters, pn, plot=True, marker='o'):
    """
    Plots the dataset with the cluster centers highlighted in red (the others white)
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param clusterCenters: list - list with the coordinates in the matrix of the cluster centers
    :param pn: integer - the number of partitions on columns and rows
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """

    if plot:
        fig = plt.figure()
        plt.title(title)

        labels = np.zeros(len(X))
        for i in range(len(X)):
            for c in range(len(clusterCenters)):
                if math.floor(X[i, 0]) == clusterCenters[c][0] and math.floor(X[i, 1]) == clusterCenters[c][1]:
                    labels[i] = 1
        label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
        ax = fig.gca()
        ax.set_xticks(np.arange(0, pn, 1))
        ax.set_yticks(np.arange(0, pn, 1))
        plt.grid(True)
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k')


def plot_spikes(spikes, step=5, title="", path='./figures/spikes_on_cluster/', save=False, show=True, ):
    """"
    Plots spikes from a simulation
    :param spikes: matrix - the list of spikes in a simulation
    :param title: string - the title of the plot
    """
    plt.figure()
    for i in range(0, len(spikes), step):
        plt.plot(np.arange(len(spikes[i])), spikes[i])
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title(title)
    if save:
        plt.savefig(path+title)
    if show:
        plt.show()


def spikes_per_cluster(spikes, labels, sim_nr):
    print("Spikes:" + str(spikes.shape))

    pca2d = PCA(n_components=2)

    for i in range(np.amax(labels) + 1):
        spikes_by_color = spikes[labels == i]
        for j in range(0, len(spikes_by_color), 20):
            plt.plot(np.arange(79), spikes_by_color[j])
        plt.title("Cluster %d Sim_%d" % (i, sim_nr))
        plt.savefig('figures/spikes_on_cluster/Sim_%d_Cluster_%d' % (sim_nr, i))
        plt.show()
        cluster_pca = pca2d.fit_transform(spikes_by_color)
        # plot(title="GT with PCA Sim_%d" % sim_nr, X=cluster_pca, marker='o')
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], c=cs.LABEL_COLOR_MAP[i], marker='o', edgecolors='k')
        plt.title("Cluster %d Sim_%d" % (i, sim_nr))
        plt.savefig('figures/spikes_on_cluster/Sim_%d_Cluster_%d_pca' % (sim_nr, i))
        plt.show()
        # print(cluster_pca)


"""
def make_distribution_one_feature(sim_nr, feature_nr, bins):
    X, _ = ds.get_dataset_simulation(sim_nr)
    spikes = X[:, feature_nr]

    sns.distplot(spikes, hist=True, kde=True,
                 bins=bins, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    # Plot formatting
    # plt.legend(prop={'size': 16}, title='Airline')
    plt.title('Distribution of spikes by magnitude for feature %s' % feature_nr)
    plt.xlabel('Magnitude')
    plt.ylabel('Density')
    plt.show()
"""


def make_distribution_all_features(sim_nr):
    X, _ = ds.get_dataset_simulation(sim_nr)
    # feature = X[:, feature_nr]
    # plot_magnitude_distribution(feature)

    for i in range(79):
        # Subset to the airline
        subset = X[:, i]

        # Draw the density plot
        sns.distplot(subset, hist=False, kde=True,
                     kde_kws={'linewidth': 2},
                     label=i)

    # Plot formatting
    plt.legend(prop={'size': 5}, title='Spikes')
    plt.title('Distribution of spikes by magnitude')
    plt.xlabel('Magnitude')
    plt.ylabel('Density')
    plt.show()


def plot_distribution(X, feature_nr, bins):
    spikes = X[:, feature_nr]

    sns.distplot(spikes, hist=True, kde=True,
                 bins=bins, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

    # Plot formatting
    # plt.legend(prop={'size': 16}, title='Airline')
    plt.title('Distribution of spikes by magnitude for feature %s' % feature_nr)
    plt.xlabel('Magnitude')
    plt.ylabel('Density')


def make_distribution_one_feature(X, feature_nr):
    spikes = X[:, feature_nr]

    distribution = np.histogram(spikes, bins=40)
    values = distribution[0]

    return values


def compute_maxima(values):
    peaks, _ = find_peaks(values, prominence=20)
    return len(peaks)


def make_distributions(X, number_of_features):
    features_distributions = []
    print("Dataset features " + str(X.shape[1]))

    for i in range(X.shape[1]):
        values = make_distribution_one_feature(X, i)
        tops = compute_maxima(values)
        features_distributions.append((i, values, tops))

    features_distributions.sort(key=lambda tup: tup[2], reverse=True)
    top_features = features_distributions[:number_of_features]
    nr_peaks = [i[2] for i in top_features]

    indexes_list = [i[0] for i in top_features]

    for i in range(0, number_of_features):
        print('Number of recorded peaks ', top_features[i][2])
        print('Feature number ', top_features[i][0])
        # plt.figure(1+i*2)
        # plt.plot(top_features[i][1])

        # plt.figure(i*2+2)
        # plot_distribution(X, top_features[i][0], 40)
        # plt.show()

    return X[:, indexes_list], nr_peaks

# # intre 23-35 aprox
# if __name__ == "__main__":
#     X, _ = ds.get_dataset_simulation(20)
#     print(X.shape)
#     X_filtered = make_distributions(X, 15)
#     print(X_filtered.shape)


