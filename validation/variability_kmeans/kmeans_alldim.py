import os

import numpy as np

from dataset_parsing import simulations_dataset as ds
from validation.performance import compute_metrics_by_kmeans

os.chdir("../")

# all_metrics = [['Simulation', 'ARI', 'AMI', 'V-Measure', 'FMI', 'DBS', 'CHS', 'SS']]
all_metrics = []
for SIM_NR in [1, 4, 16, 35]:
    spikes, labels = ds.get_dataset_simulation(simNr=SIM_NR, align_to_peak=2)
    print(spikes.shape)
    metrics = [SIM_NR]
    metrics.extend(compute_metrics_by_kmeans(spikes, labels))
    all_metrics.append(metrics)
all_metrics = np.array(all_metrics)

np.savetxt(f"./validation_kmeans/kmeans_alldim.csv", all_metrics, fmt="%.3f", delimiter=",")