import numpy as np

def spike_scaling_min_max(spikes, min_peak, max_peak):
    spikes_std = np.zeros_like(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = (spikes[:, col] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scaling_ignore_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    for row in range(len(spikes)):
        min_peak = np.amin(spikes[row])
        max_peak = np.amax(spikes[row])
        spikes_std[row] = (spikes[row] - min_peak) / (max_peak - min_peak)

    return spikes_std


def spike_scale_largest_amplitude(spikes):
    spikes_std = np.zeros_like(spikes)
    max_peak = np.amax(spikes)
    for col in range(len(spikes[0])):
        spikes_std[:, col] = spikes[:, col] / max_peak

    return spikes_std


def get_spike_energy(spike):
    return np.sum(np.power(spike, 2), axis=1)