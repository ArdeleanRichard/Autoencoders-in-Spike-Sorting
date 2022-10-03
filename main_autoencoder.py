import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils import shuffle
import os

from feature_extraction.autoencoder import lstm_input, fft_input
from feature_extraction.autoencoder.autoencoder_pca_principles.autoencoder_pca_model import AutoencoderPCAModel
from feature_extraction.autoencoder.autoencoder_tied import AutoencoderTiedModel
from feature_extraction.autoencoder.autoencoder_tied2 import AutoencoderTied2Model
from feature_extraction.autoencoder.lstm_autoencoder import LSTMAutoencoderModel
from feature_extraction.autoencoder.lstm_input import lstm_get_codes
from parameters import MODEL_PATH, PLOT_PATH
from run_utils import choose_scale, get_type
from utils.dataset_parsing import simulations_dataset as ds
from utils import scatter_plot
from feature_extraction.autoencoder.model_auxiliaries import verify_output, get_codes, get_reconstructions, \
    verify_output_one
from feature_extraction.autoencoder.autoencoder import AutoencoderModel
from feature_extraction.autoencoder.scaling import spike_scaling_min_max, spike_scaling_ignore_amplitude, get_spike_energy

from utils.dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
from utils.dataset_parsing.realdata_parsing import read_timestamps, read_waveforms, read_event_timestamps, \
    read_event_codes
from utils.dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def run_autoencoder(data_type, simulation_number, data, labels, gt_labels, index, ae_type, ae_layers, code_size, output_activation, loss_function, scale, shuff=True, noNoise=False, nr_epochs=100, doPlot=False, dropout=0.0, learning_rate=0.001, verbose=1):
    if data_type == "real":
        spikes = np.array(data[index])
        labels = np.array(labels[index])
    if data_type == "m0":
        spikes = np.array(data)
    elif data_type == "sim":
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=2)

    # print(spikes.shape)

    amplitudes = np.amax(spikes, axis=1)
    input_dim = spikes.shape[1]

    if noNoise == True:
        spikes = spikes[labels != 0]
        if data_type != "m0":
            labels = labels[labels!=0]

    if shuff == True:
        if data_type == "m0":
            spikes = shuffle(spikes, random_state=None)
        elif data_type != "real":
            spikes, labels = shuffle(spikes, labels, random_state=None)
        else:
            spikes, labels, gt_labels = shuffle(spikes, labels, gt_labels, random_state=None)

    spikes = choose_scale(spikes, scale)

    if ae_type == "shallow":
        ae_layers = [40,20,10]
        # ae_layers = [60,40,20]
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = get_codes(spikes, encoder)
    if ae_type == "normal":
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = get_codes(spikes, encoder)
    elif ae_type == "tied":
        autoencoder = AutoencoderTiedModel( input_size=len(spikes[0]),
                                            encoder_layer_sizes=ae_layers,
                                            decoder_layer_sizes=ae_layers,
                                            code_size=code_size,
                                            output_activation=output_activation,
                                            loss_function=loss_function,
                                            dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose)

        # autoencoder.save_weights(
        #     MODEL_PATH + f'autoencoder_tied_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH+"tied/")
        autoencoder_features = get_codes(spikes, encoder)
    if ae_type == "contractive":
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function="contractive",
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = get_codes(spikes, encoder)
    elif ae_type == "tied2":
        autoencoder = AutoencoderTied2Model(input_size=len(spikes[0]),
                                            encoder_layer_sizes=ae_layers,
                                            decoder_layer_sizes=ae_layers,
                                            code_size=code_size,
                                            output_activation=output_activation,
                                            loss_function=loss_function)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=0)

        # autoencoder.save_weights(
        #     MODEL_PATH + f'autoencoder_tied_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH+"tied/")
        autoencoder_features = get_codes(spikes, encoder)
    elif ae_type == "orthogonal":
        model = AutoencoderPCAModel(input_dim, ae_layers, code_size, output_activation, loss_function)
        model.train(spikes, epochs=nr_epochs)

        encoder, autoencoder = model.return_encoder()
        autoencoder_features = encoder.predict(spikes)
    elif ae_type == "ae_pca":
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        ae_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(ae_features)
    elif ae_type == "ae_pt":
        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)

        ae_layers = ae_layers.tolist()
        ae_layers.append(code_size)
        layer_weights = autoencoder.pre_train(spikes, ae_layers, epochs=100)
        autoencoder.set_weights(layer_weights)

        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = get_codes(spikes, encoder)
    elif ae_type == "lstm":
        timesteps=20
        spikes_lstm = lstm_input.temporalize_spikes(spikes, timesteps, overlap=0)
        autoencoder = LSTMAutoencoderModel(input_size=spikes_lstm.shape,
                                       lstm_layer_sizes=[60,40,20],
                                       code_size=2)

        autoencoder.train(spikes_lstm, epochs=nr_epochs, verbose=verbose)

        autoencoder.save_weights(
            MODEL_PATH + f'lstm_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        # verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = lstm_input.lstm_get_codes(spikes_lstm, encoder, timesteps)
    if ae_type == "fft":
        fft_real, fft_imag = fft_input.apply_fft_on_data(spikes, "original")

        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type("real", fft_real, fft_imag)

        spikes = np.array(spikes)

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)

        autoencoder.save_weights(
            MODEL_PATH + f'autoencoder_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
        # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
        encoder, autoencoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
        autoencoder_features = get_codes(spikes, encoder)
    if ae_type == "wfft":
        fft_real, fft_imag = fft_input.apply_fft_windowed_on_data(spikes, "blackman")

        fft_real = np.array(fft_real)
        fft_imag = np.array(fft_imag)

        spikes = get_type("magnitude", fft_real, fft_imag)

        spikes = np.array(spikes)

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=ae_layers,
                                       decoder_layer_sizes=ae_layers,
                                       code_size=code_size,
                                       output_activation=output_activation,
                                       loss_function=loss_function,
                                       dropout=dropout)
        autoencoder.train(spikes, epochs=nr_epochs, verbose=verbose, learning_rate=learning_rate)

        encoder, autoencoder = autoencoder.return_encoder()

        autoencoder_features = get_codes(spikes, encoder)

    # print(autoencoder_features.shape)
    if doPlot == True:
        if code_size == 2 or code_size == 3:
            scatter_plot.plot(f'Sim4 - GT - AE: {ae_type}', autoencoder_features, labels, marker='o')
            plt.savefig(PLOT_PATH + f'realdata_model_e{index}_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

            if scale == 'divide_amplitude':
                # ADD AMPLITUDE AFTER SCALE WITHOUT AMPL
                amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
                autoencoder_features_add_amplitude = np.hstack((autoencoder_features, amplitudes))
                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features_add_amplitude, labels, marker='o')
                plt.savefig(PLOT_PATH + f'realdata_model_e{index}_cs3+ampl_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

            plt.show()
        if code_size > 3:
            pca_2d = PCA(n_components=2)
            autoencoder_features_2d = pca_2d.fit_transform(autoencoder_features)

            kmeans_labels = KMeans(n_clusters=len(np.unique(labels))).fit_predict(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features_2d)), autoencoder_features_2d, labels, marker='o')
            plt.savefig(PLOT_PATH + f'gt_model_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

            scatter_plot.plot('GT' + str(len(autoencoder_features_2d)), autoencoder_features_2d, kmeans_labels,
                              marker='o')
            plt.savefig(PLOT_PATH + f'gt_model_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}_km')

            if scale == 'divide_amplitude':
                # ADD AMPLITUDE AFTER SCALE WITHOUT AMPL
                amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
                autoencoder_features_add_amplitude = np.hstack((autoencoder_features, amplitudes))
                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features_add_amplitude, labels, marker='o')
                plt.savefig(PLOT_PATH + f'gt_model_cs3+ampl_oa-{output_activation}_ls-_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

    # print(f"{davies_bouldin_score(autoencoder_features, labels):.3f}")
    # print(f"{calinski_harabasz_score(autoencoder_features, labels):.3f}")
    # print(f"{silhouette_score(autoencoder_features, labels):.3f}")

    return autoencoder_features, labels, gt_labels

    # pn = 25
    # sbm_labels = SBM.best(autoencoder_features, pn, ccThreshold=5, version=2)
    # scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, sbm_labels, marker='o')
    # plt.savefig(PLOT_PATH + f'gt_model_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}_sbm')




    # PCA result
    # pca_2d = PCA(n_components=2)
    # pca_features = pca_2d.fit_transform(spikes)
    # scatter_plot.plot('GT' + str(len(pca_features)), pca_features, labels, marker='o')
    # plt.savefig(PLOT_PATH + f'gt_pca_sim{simulation_number}')
    #
    # PCA on reconstructions
    # autoencoder_model = autoencoder.return_autoencoder()
    # autoencoder_reconstructions = get_reconstructions(spikes, autoencoder_model)
    #
    # pca_2d = PCA(n_components=2)
    # pca_reconstruction_features = pca_2d.fit_transform(autoencoder_reconstructions)
    #
    # scatter_plot.plot('GT' + str(len(pca_reconstruction_features)), pca_reconstruction_features, labels, marker='o')
    # plt.savefig(PLOT_PATH + f'gt_pca_reconstruction_sim{simulation_number}')

    # PCA on reconstructions * amplitude
    # autoencoder_reconstructions = get_reconstructions(spikes, autoencoder)
    # autoencoder_reconstructions = autoencoder_reconstructions * amplitudes
    #
    # pca_2d = PCA(n_components=2)
    # pca_reconstruction_features = pca_2d.fit_transform(autoencoder_reconstructions)
    #
    # scatter_plot.plot('GT' + str(len(pca_reconstruction_features)), pca_reconstruction_features, labels, marker='o')
    # plt.savefig(PLOT_PATH + f'gt_pca_reconstruction_sim{simulation_number}')





SIM_NR = 4
EPOCHS = 100
LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
ae_type = 'normal'
run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                data=None, labels=None, gt_labels=None, index=None,
                ae_type="normal", ae_layers=np.array(LAYERS), code_size=2,
                output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                doPlot=True)
run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                data=None, labels=None, gt_labels=None, index=None,
                ae_type="tied", ae_layers=np.array(LAYERS), code_size=2,
                output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                doPlot=True)




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


def compute_real_metrics(features, k):
    try:
        kmeans_labels1 = KMeans(n_clusters=k).fit_predict(features)
        kmeans_labels1 = np.array(kmeans_labels1)

        metrics = []
        metrics.append(davies_bouldin_score(features, kmeans_labels1))
        metrics.append(calinski_harabasz_score(features, kmeans_labels1))
        metrics.append(silhouette_score(features, kmeans_labels1))
    except ValueError:
        metrics = [0,0,0]
        kmeans_labels1 = np.zeros((len(features),))

    return metrics, kmeans_labels1


def get_M045_009():
    DATASET_PATH = './datasets/M045_0009/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 58

    timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)

    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)

    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=33)

    return units_in_channels, labels



# RUN REAL
# for index in [4, 6, 17, 26]:
#     for ae_type in [ "shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
#         print(f"AE{ae_type}")
#         metrics = []
#         for i in range(1, 10):
#             print(i)
#             units_in_channel, labels = get_M045_009()
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








