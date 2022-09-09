from scipy.fftpack import fft
from scipy.signal import windows
from scipy.signal.windows import gaussian
import numpy as np

from utils.dataset_parsing import simulations_dataset_autoencoder as ds

"""
Scipy FFT implementation is faster if array is of length power of 2
=> padding with/without rolling
=> reducing by deletion
"""
def apply_fft_on_range(case, alignment, range_min, range_max):
    spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True, alignment=alignment)

    # ORIGINAL SPIKE
    if case == "original":
        fft_real, fft_imag = fft_original_spike(spikes)
    # PADDED SPIKE
    elif case == "padded":
        fft_real, fft_imag = fft_padded_spike(spikes)
    # ROLLED SPIKE (also padded before)
    elif case == "rolled":
        fft_real, fft_imag = fft_rolled_spike(spikes)
    elif case == "reduced":
        fft_real, fft_imag = fft_reduced_spike(spikes)

    return fft_real, fft_imag

def apply_fft_on_data(spikes, case):
    # ORIGINAL SPIKE
    if case == "original":
        fft_real, fft_imag = fft_original_spike(spikes)
    # PADDED SPIKE
    elif case == "padded":
        fft_real, fft_imag = fft_padded_spike(spikes)
    # ROLLED SPIKE (also padded before)
    elif case == "rolled":
        fft_real, fft_imag = fft_rolled_spike(spikes)
    elif case == "reduced":
        fft_real, fft_imag = fft_reduced_spike(spikes)

    return fft_real, fft_imag

def apply_fft_windowed_on_range(alignment, range_min, range_max, window_type):
    spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True, alignment=alignment)

    if window_type == "blackman":
        windowed_spikes = apply_blackman_window(spikes)
    # std value from 5-15 to get 68% in mean-std:mean+std
    elif window_type == "gaussian":
        windowed_spikes = apply_gaussian_window(spikes, std=10)
    elif window_type == "dpss":
        M = 79
        NW = 4
        nr_windows = 5
        win, eigvals = windows.dpss(M, NW, nr_windows, return_ratios=True)

        # window = matrix with each row a window
        # spikes = matrix with each row a spike
        # element wise multiplication, row by row
        # first spike gets each window
        windowed_spikes = np.einsum('ij,kj->ikj', spikes, win)
        windowed_spikes = np.reshape(windowed_spikes, (nr_windows*len(spikes), M))

    # PADDED SPIKE
    fft_real, fft_imag = fft_padded_spike(windowed_spikes)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    return fft_real, fft_imag, labels

def apply_fft_windowed_on_data(spikes, window_type):
    if window_type == "blackman":
        windowed_spikes = apply_blackman_window(spikes)
    # std value from 5-15 to get 68% in mean-std:mean+std
    elif window_type == "gaussian":
        windowed_spikes = apply_gaussian_window(spikes, std=10)
    elif window_type == "dpss":
        M = 79
        NW = 4
        nr_windows = 5
        win, eigvals = windows.dpss(M, NW, nr_windows, return_ratios=True)

        # window = matrix with each row a window
        # spikes = matrix with each row a spike
        # element wise multiplication, row by row
        # first spike gets each window
        windowed_spikes = np.einsum('ij,kj->ikj', spikes, win)
        windowed_spikes = np.reshape(windowed_spikes, (nr_windows*len(spikes), M))

    # PADDED SPIKE
    fft_real, fft_imag = fft_padded_spike(windowed_spikes)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    return fft_real, fft_imag

def apply_fft_windowed_on_sim(sim_nr, alignment, window_type):
    spikes, labels = ds.get_dataset_simulation(simNr=sim_nr, align_to_peak=alignment)

    if window_type == "blackman":
        windowed_spikes = apply_blackman_window(spikes)
    # std value from 5-15 to get 68% in mean-std:mean+std
    elif window_type == "gaussian":
        windowed_spikes = apply_gaussian_window(spikes, std=10)
    elif window_type == "dpss":
        M = 79
        NW = 4
        nr_windows = 5
        win, eigvals = windows.dpss(M, NW, nr_windows, return_ratios=True)

        # window = matrix with each row a window
        # spikes = matrix with each row a spike
        # element wise multiplication, row by row
        # first spike gets each window
        windowed_spikes = np.einsum('ij,kj->ikj', spikes, win)
        windowed_spikes = np.reshape(windowed_spikes, (nr_windows*len(spikes), M))
    elif window_type == "none":
        windowed_spikes = spikes

    # PADDED SPIKE
    fft_real, fft_imag = fft_padded_spike(windowed_spikes)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    return fft_real, fft_imag, labels

def apply_fft_on_sim(sim_nr, case, alignment):
    spikes, labels = ds.get_dataset_simulation(simNr=sim_nr, align_to_peak=alignment)

    # ORIGINAL SPIKE
    if case == "original":
        fft_real, fft_imag = fft_original_spike(spikes)
    # PADDED SPIKE
    elif case == "padded":
        fft_real, fft_imag = fft_padded_spike(spikes)
    # ROLLED SPIKE (also padded before)
    elif case == "rolled":
        fft_real, fft_imag = fft_rolled_spike(spikes)
    elif case == "reduced":
        fft_real, fft_imag = fft_reduced_spike(spikes)

    return fft_real, fft_imag, labels

def fft_original_spike(spikes):
    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]

    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_fft_real')
    # plt.cla()

    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
PADDING AT END WITH 0
"""


# Compute power of two greater than or equal to `n`
def findNextPowerOf2(n):
    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1

    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1  # unset rightmost bit

    # `n` is now a power of two (less than `n`)

    # return next power of 2
    return n << 1

def fft_padded_spike(spikes):
    spikes = np.pad(spikes, ((0, 0), (0, findNextPowerOf2(len(spikes[0])) - len(spikes[0]))), 'constant')
    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]

    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
PADDING AT END WITH 0 and shifting the beginning (before amplitude) to the end
"""
def fft_rolled_spike(spikes):
    spikes = np.pad(spikes, ((0, 0), (0, 128 - len(spikes[0]))), 'constant')
    peak_ind = np.argmax(spikes, axis=1)

    spikes = [np.roll(spikes[i], -peak_ind[i]) for i in range(len(spikes))]
    spikes = np.array(spikes)

    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]
    #
    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
DELETE THE LAST 15 points (out of 79) to get 64 (because power of 2)
"""
def fft_reduced_spike(spikes):
    spikes = [spike[0:64] for spike in spikes]
    spikes = np.array(spikes)

    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]
    #
    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


def apply_blackman_window(spikes):
    return np.multiply(spikes, np.blackman(len(spikes[0])))


def apply_gaussian_window(spikes, std):
    return np.multiply(spikes, gaussian(M=len(spikes[0]), std=std))
