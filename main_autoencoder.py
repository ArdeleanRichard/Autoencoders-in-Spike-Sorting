import os

import numpy as np
from matplotlib import pyplot as plt

from autoencoder import run_autoencoder
from constants import LABEL_COLOR_MAP
from dataset_parsing.read_tins_m_data import get_tins_data
from dataset_parsing.simulations_dataset import get_dataset_simulation
from validation.performance import compute_real_metrics_by_kmeans
from visualization import scatter_plot
from visualization.plot_data import plot_spikes

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SIM_NR = 4
EPOCHS = 100
LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
ae_type = 'normal'
run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                data=None, labels=None, gt_labels=None, index=None,
                ae_type="normal", ae_layers=np.array(LAYERS), code_size=2,
                output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                doPlot=False)
run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                data=None, labels=None, gt_labels=None, index=None,
                ae_type="tied", ae_layers=np.array(LAYERS), code_size=2,
                output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                doPlot=True)







# # RUN REAL
# EPOCHS = 100
# LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
# # for index in [4, 6, 17, 26]:
# for index in [17]:
#     for ae_type in ["shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
#         print(f"AE{ae_type}")
#         metrics = []
#         for i in range(1, 10):
#             print(i)
#             units_in_channel, labels = get_tins_data()
#             spikes = units_in_channel[index-1]
#             spikes = np.array(spikes)
#
#             features, _, gt = run_autoencoder(data_type="m0", simulation_number=None,
#                             data=spikes, labels=None, gt_labels=None, index=None,
#                             ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
#                             output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
#                             doPlot=False, verbose=0, shuff=True)
#
#             if index == 6:
#                 met, klabels = compute_real_metrics_by_kmeans(features, 4)
#             if index == 17:
#                 met, klabels = compute_real_metrics_by_kmeans(features, 3)
#             if index == 26:
#                 met, klabels = compute_real_metrics_by_kmeans(features, 4)
#
#             scatter_plot.plot(f'K-Means on Real Data channel {index} containing {len(spikes)} spikes', features, klabels, marker='o')
#             plt.savefig(f"./figures/analysis2/m045_{index}_data/" + f'{ae_type}{i}_km_plot')
#
#             for c in np.unique(klabels):
#                 plot_spikes(f"Real Data Channel {index} cluster {c} of {len(spikes[klabels==c])} spikes", spikes[klabels==c], mean=True, color=LABEL_COLOR_MAP[c])
#                 plt.savefig(f"./figures/analysis2/m045_{index}_data/" + f'{ae_type}{i}_km_plot_spikes_cluster{c}')
#                 plt.clf()
#
#             metrics.append(met)
#         np.savetxt(f"./figures/analysis2/m045_{index}_data/{ae_type}.csv", np.array(metrics), fmt="%.3f", delimiter=",")
#







