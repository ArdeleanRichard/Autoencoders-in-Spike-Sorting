import os

import numpy as np

from autoencoder import run_autoencoder
from validation.performance import compute_metrics_by_kmeans

os.chdir("../")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RUNS = 100

EPOCHS = 50
LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
LAYER_CONFIGS = [
    [40],
    [40, 20],
    [40, 20, 10],
    [40, 20, 10, 5],
    [60, 40, 20, 10, 5],
    [60, 40, 30, 20, 10, 5],
    [70, 60, 50, 40, 30, 20, 10, 5],
    [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
]
ae_type = 'normal'
# for SIM_NR in [1,4,16,35]:
for SIM_NR in [4]:
    print(f"SIM{SIM_NR}")
    for LAYERS in LAYER_CONFIGS:
        metrics = []
        for i in range(1, RUNS):
            print(i)
            features, gt, _ = run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                            data=None, labels=None, gt_labels=None, index=None,
                            ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
                            output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.2, weight_init='he_uniform',
                            doPlot=False, verbose=0, shuff=True)
            #scatter_plot.plot(f'Autoencoder on Sim{SIM_NR}', features, gt, marker='o')
            #plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{ae_type}_sim{SIM_NR}_plot{i}')
            met = compute_metrics_by_kmeans(features, gt)
            metrics.append(met)
        # np.savetxt(f"./validation_ae/ae_{ae_type}_sim{SIM_NR}_variability_{RUNS}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")
        # np.savetxt(f"./validation_ae/ae_{ae_type}_depth{len(LAYERS)}_sim{SIM_NR}_variability_{RUNS}.csv", metrics, fmt="%.3f", delimiter=",")
        np.savetxt(f"./validation_ae/ae_{ae_type}_depth{len(LAYERS)}_sim{SIM_NR}_variability_{RUNS}_HU_do20.csv", metrics, fmt="%.3f", delimiter=",")

