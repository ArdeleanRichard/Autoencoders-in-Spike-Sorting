import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# SIM_NR = 4
# EPOCHS = 100
# LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
# ae_type = 'normal'
# run_autoencoder(data_type="sim", simulation_number=SIM_NR,
#                 data=None, labels=None, gt_labels=None, index=None,
#                 ae_type="normal", ae_layers=np.array(LAYERS), code_size=2,
#                 output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
#                 doPlot=True)
# run_autoencoder(data_type="sim", simulation_number=SIM_NR,
#                 data=None, labels=None, gt_labels=None, index=None,
#                 ae_type="tied", ae_layers=np.array(LAYERS), code_size=2,
#                 output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
#                 doPlot=True)




# for SIM_NR in [1,4,16,35]:
#     print(f"SIM{SIM_NR}")
#     metrics = []
#     for i in range(1, 10):
#         print(i)
#         features, gt, _ = run_autoencoder(data_type="sim", simulation_number=SIM_NR,
#                         data=None, labels=None, gt_labels=None, index=None,
#                         ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
#                         output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
#                         doPlot=False, verbose=0, shuff=True)
#         scatter_plot.plot(f'Autoencoder on Sim{SIM_NR}', features, gt, marker='o')
#         plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{ae_type}_sim{SIM_NR}_plot{i}')
#         met = compute_metrics(features, gt)
#         metrics.append(met)
#     np.savetxt(f"./feature_extraction/autoencoder/analysis/{ae_type}_sim{SIM_NR}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")





# RUN REAL
# for index in [4, 6, 17, 26]:
#     for ae_type in [ "shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
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
#                 met, klabels = compute_real_metrics(features, 4)
#             if index == 17:
#                 met, klabels = compute_real_metrics(features, 3)
#             if index == 26:
#                 met, klabels = compute_real_metrics(features, 4)
#
#             scatter_plot.plot(f'K-Means on C37', features, klabels, marker='o')
#             plt.savefig(f"./feature_extraction/autoencoder/analysis/m045_{index}_data/" + f'{ae_type}_m045_{index}_km_plot{i}')
#
#             metrics.append(met)
#         np.savetxt(f"./feature_extraction/autoencoder/analysis/m045_{index}_data/{ae_type}_m045_{index}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")








