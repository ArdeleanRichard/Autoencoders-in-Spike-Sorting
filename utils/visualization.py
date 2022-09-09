import utils.constants as cs
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from feature_extraction.weighted_pca.feature_statistics import compute_kde, find_kde_bw

def plot_clusters(title, X, labels=None, marker='o', plot=True, save=False, filename=""):
    """
    Plots and optionally saves the plot of the FIRST two dimensions of the dataset X.

    @param title: string, title of the plot
    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param labels: 1D array, clustering label of each data sample
    @param marker: type of the plotting marker
    @param plot: Boolean, True - show plot
    @param save: Boolean, True - save plot
    @param filename: string, location to save the plot
    """

    plt.figure()
    plt.title(title)

    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], marker=marker, edgecolors='k')
    else:
        try:
            label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
            plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', alpha=1, edgecolors='k')
        except KeyError:
            print('Too many labels! Using default colors...\n')

        plt.ylabel('Magnitude (mV)')
        plt.xlabel('Magnitude (mV)')

        if save:
            plt.savefig(filename)
        if plot:
            plt.show()

        plt.close()


def plot_more_dims(X, labels, save=False, filename=""):
    """
    Plots and optionally saves the dataset X, creating a matrix of shape (n_features, n_features) of plots.
    Each matrix cell represents a plot of two dimensions.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param labels: 1D array, clustering label of each data sample
    @param save: Boolean, True - save the plot
    @param filename: string, location to save the plot
    """

    df = pd.DataFrame(X)
    df['cluster'] = labels
    sns.pairplot(df, diag_kind='kde', hue='cluster', height=3)

    if save:
        plt.savefig(filename)

    plt.show()


def plot_distribution_one_feature(X, feature_nr, hist=False, save=False, filename=""):
    """
    Plots the histogram or the kernel density estimator of a feature from dataset X.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param feature_nr: integer, index of the feature used for plotting
    @param hist: Boolean, True - plot histogram, False - plot kernel density estimator
    @param save: Boolean, True - save the plot
    @param filename: string, location to save the plot
    """

    spikes = X[:, feature_nr]

    if hist:
        sns.displot(spikes, kde=False, discrete=False, element="bars", stat="count", common_norm=False)
    else:
        sns.displot(spikes, kind="kde")

    plt.xlabel('magnitude (mV)')

    if hist:
        plt.ylabel('count')
    else:
        plt.ylabel('probability density')

    if save:
        plt.savefig(filename)
    # plt.show()


def two_features_plot(X, f1, bw1, f2, bw2, sim):
    """
    Plot the kde of two features in the same plot.

    @param X: 2D array of shape (n_samples, n_features), stores the data
    @param f1: integer, index of the first feature
    @param bw1: bandwidth of kde of feature 1
    @param f2: integer, index of the second feature
    @param bw2: bandwidth of kde of feature 2
    @param sim: simulation number
    """

    x1 = np.linspace(X[:, f1].min(), X[:, f1].max(), X[:, f1].shape[0])[:, np.newaxis]
    kde1 = compute_kde(X[:, f1], bw1)
    pdf1 = np.exp(kde1.score_samples(x1))
    label = 'f' + str(f1)
    dc1, = plt.plot(x1, pdf1, label=label)

    x2 = np.linspace(X[:, f2].min(), X[:, f2].max(), X[:, f2].shape[0])[:, np.newaxis]
    kde2 = compute_kde(X[:, f2], bw2)
    pdf2 = np.exp(kde2.score_samples(x2))
    label = 'f' + str(f2)
    dc2, = plt.plot(x2, pdf2, label=label)

    plt.xlabel('magnitude', fontsize=15)
    plt.ylabel('density', fontsize=15)
    plt.legend(handles=[dc1, dc2])

    fig_name = 's' + str(sim) + 'f' + str(f1) + 'f' + str(f2)
    plt.savefig(fig_name)

    plt.show()


def plot_two_distributions(p, q, figname=""):
    """
    Plot the kde of two distributions in the same plot
    """

    bw1 = find_kde_bw(p)
    bw2 = find_kde_bw(q)

    x1 = np.linspace(p.min(), p.max(), p.shape[0])[:, np.newaxis]
    kde1 = compute_kde(p, bw1)
    pdf1 = np.exp(kde1.score_samples(x1))
    label = 'before'
    dc1, = plt.plot(x1, pdf1, label=label)

    x2 = np.linspace(q.min(), q.max(), q.shape[0])[:, np.newaxis]
    kde2 = compute_kde(q, bw2)
    pdf2 = np.exp(kde2.score_samples(x2))
    label = 'after'
    dc2, = plt.plot(x2, pdf2, label=label)

    plt.xlabel('magnitude', fontsize=15)
    plt.ylabel('density', fontsize=15)
    plt.legend(handles=[dc1, dc2])

    plt.title = "before/after weighting f" + figname
    plt.savefig(figname)
    plt.close()

    # plt.show()


def one_feature_statistics(f1, figname):
    from feature_extraction.weighted_pca.feature_statistics import compute_kde, find_kde_bw

    bw1 = find_kde_bw(f1)

    x1 = np.linspace(f1.min(), f1.max(), len(f1))[:, np.newaxis]
    kde1 = compute_kde(f1, bw1)
    pdf1 = np.exp(kde1.score_samples(x1))
    label = 'old'
    dc1, = plt.plot(x1, pdf1, label=label)

    plt.xlabel('magnitude')
    plt.ylabel('density')
    plt.legend(handles=[dc1])

    old_min = round(f1.min(), 3)
    old_max = round(f1.max(), 3)
    old_var = round(f1.var(), 3)

    textstr = 'range: [%.3f, %.3f]\nnew var:%.3f' % (old_min, old_max, old_var)

    plt.text(old_max + 0.1, 0, textstr, fontsize=8)
    # plt.title(title)
    # plt.grid(True)
    plt.subplots_adjust(right=0.75)

    # if os.path.isdir(folder_name) is False:
    #     os.mkdir(folder_name)
    #
    # plt.savefig(folder_name + '/' + title)
    # plt.close()
    #
    plt.savefig(figname)
    # plt.show()
    plt.close()


def two_features_statistics(f1, f2, title, folder_name, w):
    """
    Plot the kde of two features in the same plot
    """

    bw1 = find_kde_bw(f1)
    bw2 = find_kde_bw(f2)

    x1 = np.linspace(f1.min(), f1.max(), len(f1))[:, np.newaxis]
    kde1 = compute_kde(f1, bw1)
    pdf1 = np.exp(kde1.score_samples(x1))
    label = 'old'
    dc1, = plt.plot(x1, pdf1, label=label)

    x2 = np.linspace(f2.min(), f2.max(), len(f2))[:, np.newaxis]
    kde2 = compute_kde(f2, bw2)
    pdf2 = np.exp(kde2.score_samples(x2))
    label = 'new'
    dc2, = plt.plot(x2, pdf2, label=label)

    plt.xlabel('magnitude')
    plt.ylabel('density')
    plt.legend(handles=[dc1, dc2])

    # fig_name = 's'+str(sim)+'f' + str(f1) + 'f' + str(f2)
    # plt.savefig(fig_name)

    new_min = round(f2.min(), 3)
    new_max = round(f2.max(), 3)
    old_min = round(f1.min(), 3)
    old_max = round(f1.max(), 3)
    new_var = round(f2.var(), 3)
    old_var = round(f1.var(), 3)

    # print(new_min, new_max, old_min, old_max, new_var, old_var)

    textstr = 'w:%.3f\nold range: [%.3f, %.3f]\nnew range: [%.3f, %.3f]\nold var:%.3f\nnew var:%.3f' % (
        w, old_min, old_max, new_min, new_max, old_var, new_var)

    plt.text(max(old_max, new_max) + 0.1, 0, textstr, fontsize=8)
    plt.title(title)
    # plt.grid(True)
    plt.subplots_adjust(right=0.75)

    if os.path.isdir(folder_name) is False:
        os.mkdir(folder_name)

    plt.savefig(folder_name + '/' + title)
    plt.close()

    # plt.show()


def plot_heatmap(matrix, title="", figure_name=""):
    # y = np.arange(15, 50)
    # x = [0, 1]
    # sns.heatmap(matrix, xticklabels=x, yticklabels=y, annot=True, cmap="YlGnBu")

    sns.heatmap(matrix, annot=False, cmap="YlGnBu")

    plt.title(title)
    plt.xlabel("principal components")
    plt.ylabel("features")

    if figure_name != "":
        plt.savefig(figure_name)
        plt.show()


def plot_individual_clusters(X, y):
    c1 = np.zeros((202, 2))
    c2 = np.zeros((531, 2))
    c3 = np.zeros((1127, 2))

    index = 0
    index2 = 0
    index3 = 0

    for j, i in enumerate(y):
        if i == 1:
            c1[index] = X[j]
            index += 1
        if i == 2:
            c2[index2] = X[j]
            index2 += 1
        if i == 3:
            c3[index3] = X[j]
            index3 += 1

    labels = np.full(len(c1), 1)
    plot_clusters("c1", c1, labels)

    labels = np.full(len(c2), 2)
    plot_clusters("c2", c2, labels)


def plot_two_data(title, X1, X2, labels=None, marker='o', plot=True, save=False, filename=""):
    plt.figure()
    plt.title(title)

    label_color_1 = [cs.LABEL_COLOR_MAP[2] for i in np.arange(1, 10)]
    label_color_2 = [cs.LABEL_COLOR_MAP[6] for i in np.arange(1, 10)]

    # first
    plt.scatter(X1[:9, 0], X1[:9, 1], c=label_color_1, marker='o', alpha=1, edgecolors='k')
    plt.scatter(X1[9:18, 0], X1[9:18, 1], c=label_color_1, marker='x', alpha=1, edgecolors='k')

    # second
    plt.scatter(X2[:9, 0], X2[:9, 1], c=label_color_2, marker='o', alpha=1, edgecolors='k')
    plt.scatter(X2[9:18, 0], X2[9:18, 1], c=label_color_2, marker='x', alpha=1, edgecolors='k')

    # draw lines between them
    for i in range(9):
        # plt.plot([X1[i][0], X2[i][0]], [X1[i][1], X2[i][1]], c='lightblue')
        plt.annotate("",
                     xy=(X2[i][0], X2[i][1]), xycoords='data',
                     xytext=(X1[i][0], X1[i][1]), textcoords='data',
                     arrowprops=dict(arrowstyle="->",
                                     color="0.5",
                                     shrinkA=5, shrinkB=5,
                                     patchA=None,
                                     patchB=None,
                                     ),
                     )

    for i in range(9, 18):
        # plt.plot([X1[i, 0], X2[i][0]], [X1[i, 1], X2[i][1]], c='lavender')
        plt.annotate("",
                     xy=(X2[i][0], X2[i][1]), xycoords='data',
                     xytext=(X1[i][0], X1[i][1]), textcoords='data',
                     arrowprops=dict(arrowstyle="->",  # linestyle="dashed",
                                     color="0.5",
                                     shrinkA=5, shrinkB=5,
                                     patchA=None,
                                     patchB=None,
                                     ),
                     )

    for i_x, i_y in zip(X1[:, 0], X1[:, 1]):
        plt.text(i_x, i_y, '({}, {})'.format(round(i_x, 2), round(i_y, 2)), fontsize="x-small")

    for i_x, i_y in zip(X2[:, 0], X2[:, 1]):
        plt.text(i_x, i_y, '({}, {})'.format(round(i_x, 2), round(i_y, 2)), fontsize="x-small")

    plt.ylabel('mV')
    plt.xlabel('mV')

    if save:
        plt.savefig(filename)
    if plot:
        plt.show()

    plt.close()
