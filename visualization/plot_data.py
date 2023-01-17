import numpy as np
from matplotlib import pyplot as plt

from utils.constants import LABEL_COLOR_MAP


def plot_spikes(title, spikes, limit=False, mean=False, color='blue'):
    plt.title(title)
    for spike in spikes:
        if mean == True:
            plt.plot(spike, 'gray')
        else:
            plt.plot(spike)
    if mean == True:
        plt.plot(np.mean(spikes, axis=0), color)
    if limit == True:
        plt.ylim(-120, 80)


def plot_spikes_by_clusters(spikes, labels, mean=True, title=''):
    for lab in np.unique(labels):
        plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        for spike in spikes[labels==lab]:
            if mean == True:
                plt.plot(spike, 'gray')
            else:
                plt.plot(spike)
        if mean == True:
            plt.plot(np.mean(spikes[labels==lab], axis=0), LABEL_COLOR_MAP[lab])
        plt.savefig(f"./figures/waveforms/{title}_cluster{lab}")
    # plt.show()