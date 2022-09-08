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
from main_ardelean import get_type
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
PLOT_PATH = './feature_extraction/autoencoder/testfigs/'
MODEL_PATH = './feature_extraction/autoencoder/weights/'

simulation_number = 1
# output_activation = 'linear'
output_activation = 'tanh'
# output_activation = 'spp'

loss_function = 'mse'
# loss_function = 'bce'
# loss_function = 'ce'

# loss_function = 'cce'
# loss_function = 'scce'



def choose_scale(spikes, scale_type):
    if scale_type == 'minmax':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    if scale_type == 'minmax_relu':
        spikes = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
    if scale_type == 'minmax_spp':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 4) - 3
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == '-1+1':
        spikes = spike_scaling_min_max(spikes, min_peak=-1, max_peak=1)
        # plot_spikes(spikes_scaled, title='scale_-1_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'scaler':
        spikes = preprocessing.MinMaxScaler((0, 1)).fit_transform(spikes)
        # plot_spikes(spikes_scaled, title='scale_sklearn_0_1', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude':
        # SCALE IGNORE AMPLITUDE
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_no_amplitude', path=PLOT_PATH, show=False, save=True)
    elif scale_type == 'ignore_amplitude_add_amplitude':
        # SCALE IGNORE AMPLITUDE ADDED FEATURE AMPLITUDE
        amplitudes = np.amax(spikes, axis=1)
        amplitudes = amplitudes.reshape((-1,1))
        spikes_scaled = spike_scaling_ignore_amplitude(spikes)
        spikes = (spikes_scaled * 2) - 1
        spikes = np.hstack((spikes, amplitudes))
    elif scale_type == 'add_energy':
        # SCALED ADDED FEATURE ENERGY
        spikes_energy = get_spike_energy(spikes)
        spikes_energy = spikes_energy.reshape((-1,1))
        spikes = np.hstack((spikes, spikes_energy))
        # print(spikes.shape)
    elif scale_type == 'divide_amplitude':
        amplitudes = np.amax(spikes, axis=1)
        print(len(amplitudes))
        amplitudes = np.reshape(amplitudes, (len(amplitudes), -1))
        spikes = spikes / amplitudes
    elif scale_type == 'scale_no_energy_loss':
        # print(spikes)
        for spike in spikes:
            spike[spike < 0] = spike[spike < 0] / abs(np.amin(spike))
            spike[spike > 0] = spike[spike > 0] / np.amax(spike)

    return spikes


#autoencoder_layer_sizes = [40, 35, 30, 25, 20, 15, 10, 5], code_size=2
def run_autoencoder_selected(simulation_number, autoencoder_layer_sizes, code_size, output_activation, loss_function, scale, shuff=True):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
    print(spikes.shape)
    nr_epochs = 100

    if shuff == True:
        spikes, labels = shuffle(spikes, labels, random_state=None)

    spikes = spikes[:, 20:60]
    print(spikes.shape)


    amplitudes = np.amax(spikes, axis=1)
    print(len(amplitudes))

    spikes = choose_scale(spikes, scale)

    input_size = len(spikes[0])
    autoencoder = AutoencoderModel(input_size=input_size,
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=code_size,
                                   output_activation=output_activation,
                                   loss_function=loss_function)
    autoencoder.train(spikes, epochs=nr_epochs)
    autoencoder.save_weights(
        MODEL_PATH + f'autoencoder{input_size}_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
    # autoencoder.load_weights('./feature_extraction/autoencoder/weights/autoencoder_cs{code_size}_oa-tanh_sim4_e100')
    encoder, autoencoder = autoencoder.return_encoder()

    verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
    autoencoder_features = get_codes(spikes, encoder)

    print(autoencoder_features.shape)

    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
    plt.savefig(
        PLOT_PATH + f'gt_model{input_size}_cs{code_size}_scale{scale}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

    return autoencoder_features
# run_autoencoder_selected(simulation_number, output_activation, loss_function, scale='minmax', shuff=True)


def run_orthogonal_autoencoder(activation):
    spikes, labels = ds.get_dataset_simulation(simNr=4)
    spikes, labels = shuffle(spikes, labels, random_state=None)
    spikes = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
    #spikes = (spikes * 2) - 1

    input_dim = spikes.shape[1]  # num of predictor variables,
    # activation = 'tanh'
    loss = 'mse'
    model = AutoencoderPCAModel(input_dim, 2, activation, loss)
    model.train(spikes, epochs=100)

    encoder, autoencoder = model.return_encoder()

    train_predictions = autoencoder.predict(spikes)
    print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(spikes, train_predictions))

    code_predictions = encoder.predict(spikes)
    print(code_predictions)

    scatter_plot.plot('GT' + str(len(code_predictions)), code_predictions, labels, marker='o')

    plt.figure()
    plt.plot(np.arange(len(spikes[0])), spikes[0], label="original")
    plt.plot(np.arange(len(train_predictions[0])), train_predictions[0], label="reconstructed")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike 0")
    plt.show()
# run_orthogonal_autoencoder('linear')
# run_orthogonal_autoencoder('spp')
# run_orthogonal_autoencoder('tanh')


def combine_autoencoders(simulation_number, loss_function, shuff=True):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

    if shuff == True:
        spikes, labels = shuffle(spikes, labels, random_state=None)
    output_activation = 'tanh'
    ae_layers = [70,60,50,40,30,20,10,5]
    code_size = 2
    ae_features1 = run_autoencoder(simulation_number, ae_layers, code_size, output_activation, loss_function, scale='')
    ae_features2 = run_autoencoder(simulation_number, ae_layers, code_size, output_activation, loss_function, scale='minmax')
    output_activation = 'spp'
    ae_features3 = run_autoencoder(simulation_number, ae_layers, code_size, output_activation, loss_function, scale='')
    ae_features4 = run_autoencoder(simulation_number, ae_layers, code_size, output_activation, loss_function, scale='minmax_spp')


    ae_features1 = preprocessing.MinMaxScaler((0, 1)).fit_transform(ae_features1)
    ae_features2 = preprocessing.MinMaxScaler((0, 1)).fit_transform(ae_features2)
    ae_features3 = preprocessing.MinMaxScaler((0, 1)).fit_transform(ae_features3)
    ae_features4 = preprocessing.MinMaxScaler((0, 1)).fit_transform(ae_features4)

    ae_features1 = ae_features1*2 + 1
    ae_features2 = ae_features2*2 + 1
    ae_features3 = ae_features3*2 + 1
    ae_features4 = ae_features4*2 + 1

    scatter_plot.plot('GT' + str(len(ae_features1)), ae_features1, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_cs{code_size}_ls-{loss_function}_sim{simulation_number}_1')
    scatter_plot.plot('GT' + str(len(ae_features2)), ae_features2, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_cs{code_size}_ls-{loss_function}_sim{simulation_number}_2')
    scatter_plot.plot('GT' + str(len(ae_features3)), ae_features3, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_cs{code_size}_ls-{loss_function}_sim{simulation_number}_3')
    scatter_plot.plot('GT' + str(len(ae_features4)), ae_features4, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_cs{code_size}_ls-{loss_function}_sim{simulation_number}_4')



    autoencoder_features = ae_features1 * ae_features2 * ae_features3 * ae_features4

    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_mul_cs{code_size}_ls-{loss_function}_sim{simulation_number}')

    autoencoder_features = ae_features1 + ae_features2 + ae_features3 + ae_features4

    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_combo_add_cs{code_size}_ls-{loss_function}_sim{simulation_number}')
# combine_autoencoders(simulation_number, loss_function, shuff=True)



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
LAYERS = [70, 60, 50, 40, 30,20,10,5]
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


##### CHECK CLUSTERING METRICS AND FE METRICS
def compute_retard_metrics(components, scaling, features, gt_labels):
    try:
        kmeans_labels1 = KMeans(n_clusters=len(np.unique(gt_labels))).fit_predict(features)
        kmeans_labels1 = np.array(kmeans_labels1)
        gt_labels = np.array(gt_labels)
        # kmeans_labels2 = KMeans(n_clusters=2).fit_predict(features)

        # print("Full Labeled")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {adjusted_rand_score(kmeans_labels1, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {adjusted_mutual_info_score(kmeans_labels1, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {v_measure_score(kmeans_labels1, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {fowlkes_mallows_score(kmeans_labels1, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {davies_bouldin_score(features, kmeans_labels1):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {calinski_harabasz_score(features, kmeans_labels1):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:   {silhouette_score(features, kmeans_labels1):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:     {davies_bouldin_score(features, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:     {calinski_harabasz_score(features, labels):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:     {silhouette_score(features, labels):.3f}")

        # print("Ground Truth")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {adjusted_rand_score(kmeans_labels2, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {adjusted_mutual_info_score(kmeans_labels2, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {v_measure_score(kmeans_labels2, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {fowlkes_mallows_score(kmeans_labels2, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {davies_bouldin_score(features, kmeans_labels2):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {calinski_harabasz_score(features, kmeans_labels2):.3f}")
        # print(f"C{components} -> S{scaling} -> ClustEval:  {silhouette_score(features, kmeans_labels2):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:    {davies_bouldin_score(features, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:    {calinski_harabasz_score(features, gt_labels):.3f}")
        # print(f"C{components} -> S{scaling} -> FE-Eval:    {silhouette_score(features, gt_labels):.3f}")
        #
        # print()
        # print()

        metrics = []
        metrics.append(adjusted_rand_score(kmeans_labels1, gt_labels))
        metrics.append(adjusted_mutual_info_score(kmeans_labels1, gt_labels))
        metrics.append(v_measure_score(kmeans_labels1, gt_labels))
        metrics.append(fowlkes_mallows_score(kmeans_labels1, gt_labels))
        metrics.append(davies_bouldin_score(features, kmeans_labels1))
        metrics.append(calinski_harabasz_score(features, kmeans_labels1))
        metrics.append(silhouette_score(features, kmeans_labels1))
        metrics.append(davies_bouldin_score(features, gt_labels))
        metrics.append(calinski_harabasz_score(features, gt_labels))
        metrics.append(silhouette_score(features, gt_labels))
    except ValueError:
        metrics = [0,0,0,0,0,0,0,0,0,0]

    return metrics


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
#         met = compute_retard_metrics(None, None, features, gt)
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


def read_real_data():
    from utils.dataset_parsing.realdata_ssd_1electrode import parse_ssd_file
    from utils.dataset_parsing.realdata_parsing import read_timestamps, read_waveforms
    from utils.dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel

    DATASET_PATH = './datasets/real_data/M017_4/sorted/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 58
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 32

    timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)

    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)

    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

    return units_in_channels, labels
# units_in_channels, labels = read_real_data()
# run_autoencoder_real_data(units_in_channels, labels, 0, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 4, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 5, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 8, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 0, output_activation, loss_function, scale='divide_amplitude', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 4, output_activation, loss_function, scale='divide_amplitude', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 5, output_activation, loss_function, scale='divide_amplitude', shuff=True)
# run_autoencoder_real_data(units_in_channels, labels, 8, output_activation, loss_function, scale='divide_amplitude', shuff=True)

def read_kampff_c37():
    DATASET_PATH = './datasets/kampff/c37/Spikes/'

    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1]
    #*# unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

    intracellular_labels = np.zeros((len(timestamps)))
    # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
    # for index, timestamp in enumerate(timestamps):
    #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
    #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
    #             given_index[index2] = 1
    #             intracellular_labels[index] = 1
    #             break

    for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        indexes = []
        for index, timestamp in enumerate(timestamps):
            if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH:
                # given_index[index2] = 1
                indexes.append(index)

        if indexes != []:
            min = indexes[0]
            for i in range(1, len(indexes)):
                if timestamps[indexes[i]] < timestamps[min]:
                    min = indexes[i]
            intracellular_labels[min] = 1

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels


def read_kampff_c28():
    DATASET_PATH = './datasets/kampff/c28/units/'
    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 54
    TIMESTAMP_LENGTH = 1
    NR_CHANNELS = 1
    unit_electrode = [1, 1, 1, 1]

    timestamp_file, waveform_file, event_timestamps_filename, event_codes_filename = find_ssd_files(DATASET_PATH)
    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)
    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)
    waveform_lens = list(map(len, waveforms_by_unit))
    event_timestamps = read_event_timestamps(event_timestamps_filename)
    event_codes = read_event_codes(event_codes_filename)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH,
                                                 number_of_channels=NR_CHANNELS)

    intracellular_labels = np.zeros((len(timestamps)))
    # given_index = np.zeros((len(event_timestamps[event_codes == 1])))
    # for index, timestamp in enumerate(timestamps):
    #     for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
    #         if event_timestamp - 1000 < timestamp < event_timestamp + 1000 and given_index[index2] == 0:
    #             given_index[index2] = 1
    #             intracellular_labels[index] = 1
    #             break

    for index2, event_timestamp in enumerate(event_timestamps[event_codes == 1]):
        indexes = []
        for index, timestamp in enumerate(timestamps):
            if event_timestamp - WAVEFORM_LENGTH < timestamp < event_timestamp + WAVEFORM_LENGTH:
                # given_index[index2] = 1
                indexes.append(index)

        if indexes != []:
            min = indexes[0]
            for i in range(1, len(indexes)):
                if timestamps[indexes[i]] < timestamps[min]:
                    min = indexes[i]
            intracellular_labels[min] = 1

    # return units_in_channels[0], labels, intracellular_labels
    return units_in_channels, labels, intracellular_labels

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




for index in [6, 17, 26]:
    for ae_type in [ "shallow", "normal", "tied", "contractive", "orthogonal", "ae_pca", "ae_pt", "lstm", "fft",  "wfft"]:
        print(f"AE{ae_type}")
        metrics = []
        for i in range(1, 10):
            print(i)
            units_in_channel, labels = get_M045_009()
            spikes = units_in_channel[index-1]
            spikes = np.array(spikes)

            features, _, gt = run_autoencoder(data_type="m0", simulation_number=None,
                            data=spikes, labels=None, gt_labels=None, index=None,
                            ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
                            output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS, dropout=0.0,
                            doPlot=False, verbose=0, shuff=True)

            if index == 6:
                met, klabels = compute_real_metrics(features, 4)
            if index == 17:
                met, klabels = compute_real_metrics(features, 3)
            if index == 26:
                met, klabels = compute_real_metrics(features, 4)

            scatter_plot.plot(f'K-Means on C37', features, klabels, marker='o')
            plt.savefig(f"./feature_extraction/autoencoder/analysis/m045_{index}_data/" + f'{ae_type}_m045_{index}_km_plot{i}')

            metrics.append(met)
        np.savetxt(f"./feature_extraction/autoencoder/analysis/m045_{index}_data/{ae_type}_m045_{index}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")



# spikes, labels, gt_labels = read_kampff_c37()
# spikes, labels, gt_labels = read_kampff_c28()
# run_autoencoder(data_type="sim", simulation_number=SIM_NR,
#                 data=None, labels=None, gt_labels=None, index=None,
#                 ae_type="tied2", ae_layers=np.array(LAYERS), code_size=2,
#                 output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS,
#                 doPlot=True)
# run_autoencoder(data_type="sim", simulation_number=SIM_NR,
#                 data=None, labels=None, gt_labels=None, index=None,
#                 ae_type="orthogonal", ae_layers=np.array(LAYERS), code_size=2,
#                 output_activation='tanh', loss_function='mse', scale="minmax", nr_epochs=EPOCHS,
#                 doPlot=True)

# run_autoencoder(simulation_number, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder(simulation_number, output_activation, loss_function, scale='scale_no_energy_loss', shuff=True)
# run_autoencoder(simulation_number, output_activation, loss_function, scale='divide_amplitude', shuff=True)
# run_autoencoder_high_code(simulation_number, output_activation, loss_function, scale='divide_amplitude', shuff=True)
# run_autoencoder_high_code(simulation_number, output_activation, loss_function, scale='divide_amplitude', shuff=True)





# run_autoencoder_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='divide_amplitude', shuff=True)

# run_autoencoder_high_code_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='no', shuff=True)
# run_autoencoder_high_code_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='minmax', shuff=True)
# run_autoencoder_high_code_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='divide_amplitude', shuff=True)

# run_orthogonal_autoencoder_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='no', shuff=True)
# run_orthogonal_autoencoder_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='minmax', shuff=True)
# run_orthogonal_autoencoder_real_data(spikes, np.array([gt_labels]), 0, output_activation, loss_function, scale='divide_amplitude', shuff=True)

def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

def real_feature_extraction():
    spikes, labels, gt_labels = read_kampff_c37()

    pca_2d = PCA(n_components=2)
    data = pca_2d.fit_transform(spikes)

    from sklearn.metrics import silhouette_score

    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))

    print(sil)
    test = calculate_WSS(data, 10)
    print(test)




def run_autoencoder_cascaded(simulation_number, autoencoder_layer_sizes, code_size, output_activation, loss_function, scale, shuff=True):
    spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
    print(spikes.shape)
    nr_epochs = 100

    if shuff == True:
        spikes, labels = shuffle(spikes, labels, random_state=None)

    if scale == 'minmax':
        # plot_spikes(spikes, title='scale_no', path=PLOT_PATH, show=False, save=True)
        spikes_scaled = spike_scaling_min_max(spikes, min_peak=np.amin(spikes), max_peak=np.amax(spikes))
        # plot_spikes(spikes_scaled, title='scale_min_max', path=PLOT_PATH, show=False, save=True)
        spikes = (spikes_scaled * 2) - 1
        # plot_spikes(spikes_scaled, title='scale_mod_-1_1', path=PLOT_PATH, show=False, save=True)


    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=code_size,
                                   output_activation=output_activation,
                                   loss_function=loss_function)
    autoencoder.train(spikes, epochs=nr_epochs)
    autoencoder.save_weights(
        MODEL_PATH + f'autoencoder_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')

    encoder, autoencoder = autoencoder.return_encoder()

    verify_output(spikes, encoder, autoencoder, path=PLOT_PATH)
    autoencoder_reconstructions = get_reconstructions(spikes, autoencoder)

    print(autoencoder_reconstructions.shape)

    autoencoder_cascaded = AutoencoderModel(input_size=len(autoencoder_reconstructions[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=code_size,
                                   output_activation=output_activation,
                                   loss_function=loss_function)
    autoencoder_cascaded.train(autoencoder_reconstructions, epochs=nr_epochs)
    encoder_cascaded, autoencoder_cascaded = autoencoder_cascaded.return_encoder()

    verify_output(autoencoder_reconstructions, encoder_cascaded, autoencoder_cascaded, i=1, path=PLOT_PATH)
    autoencoder_features = get_codes(autoencoder_reconstructions, encoder_cascaded)

    scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
    plt.savefig(PLOT_PATH + f'gt_model_cascaded_cs{code_size}_oa-{output_activation}_ls-{loss_function}_sim{simulation_number}_e{nr_epochs}')
# run_autoencoder_cascaded(simulation_number, output_activation, loss_function, scale='minmax', shuff=True)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_mat, axis=0)) / np.sum(contingency_mat)


def compare_metrics(Data, X, y, n_clusters):
    # dataset

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    #metric - ARI
    print(f"{Data} - ARI: "
          f"KMeans={adjusted_rand_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - AMI: "
          f"KMeans={adjusted_mutual_info_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - Purity: "
          f"KMeans={purity_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - FMI: "
          f"KMeans={fowlkes_mallows_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - VM: "
          f"KMeans={v_measure_score(y, kmeans.labels_):.3f}\t")

    print(f"{Data} - SS: "
          f"KMeans={silhouette_score(X, kmeans.labels_):.3f}\t")

    print(f"{Data} - CHS: "
          f"KMeans={calinski_harabasz_score(X, kmeans.labels_):.3f}\t")

    print(f"{Data} - DBS: "
          f"KMeans={davies_bouldin_score(X, kmeans.labels_):.3f}\t")

    print()









def calculate_metrics_table():
    method = "PCA"
    method = "ae_normal"
    method = "ae_ortho"
    metrics_saved = []
    for components in [2, 3, 20]:
        for scaling in ["-", "minmax", "divide_amplitude"]:
            # spikes, labels, gt_labels = read_kampff_c28()
            spikes, labels, gt_labels = read_kampff_c37()
            spikes = np.array(spikes)
            labels = np.array(labels)
            if method == "PCA":
                spikes = choose_scale(spikes[0], scaling)
                pca_instance = PCA(n_components=components)
                features = pca_instance.fit_transform(spikes)
            elif method == "ae_normal":
                if components == 20:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="normal",
                                                                  ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                                  code_size=components,
                                                                  output_activation=output_activation,
                                                                  loss_function=loss_function,
                                                                  scale=scaling, shuff=True, nr_epochs=100)

                else:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="normal",
                                                                  ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                                  code_size=components,
                                                                  output_activation=output_activation,
                                                                  loss_function=loss_function,
                                                                  scale=scaling, shuff=True, nr_epochs=100)
            elif method == "ae_ortho":
                if components == 20:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="orthogonal",
                                                                  ae_layers=[60, 50, 40, 30], code_size=components,
                                                                  output_activation=output_activation, loss_function=loss_function,
                                                                  scale=scaling, shuff=True)
                else:
                    features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                                  labels=labels, gt_labels=gt_labels, index=0,
                                                                  ae_type="orthogonal",
                                                                  ae_layers=[40, 20, 10, 5], code_size=components,
                                                                  output_activation=output_activation, loss_function=loss_function,
                                                                  scale=scaling, shuff=True)

            metrics_saved.append(compute_retard_metrics(components, scaling, features, labels, gt_labels))

    np.savetxt(f"./feature_extraction/autoencoder/analysis/analysis_c37_{method}.csv", np.around(np.array(metrics_saved), decimals=3).transpose(), delimiter=",")

# calculate_metrics_table()











def plot_metrics_clustering_eval(title, pca, oae, xlabel, ylabel):
    max_saved = max(oae[5], pca[5])
    oae[5] = oae[5] / max_saved
    pca[5] = pca[5] / max_saved

    max_saved = max(oae[4], pca[4])
    oae[4] = oae[4] / max_saved
    pca[4] = pca[4] / max_saved

    plt.title(title)
    plt.plot([0, 1], [0, 1], 'g--')
    plt.scatter(pca, oae, c=[0,0,0,0,1,0,0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()



def plot_metrics_fe_eval(title, pca, oae, xlabel, ylabel):
    max_saved = max(oae[0], pca[0])
    oae[0] = oae[0] / max_saved
    pca[0] = pca[0] / max_saved

    max_saved = max(oae[1], pca[1])
    oae[1] = oae[1] / max_saved
    pca[1] = pca[1] / max_saved

    plt.title(title)
    plt.plot([-0.5, 1], [-0.5, 1], 'g--')
    plt.scatter(pca, oae, c=[1,0,0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()






def calculate_metrics_specific(spikes, labels, gt_labels, method="PCA", components=2, scaling="-"):
    spikes = np.array(spikes)
    labels = np.array(labels)
    if method == "PCA":
        spikes = choose_scale(spikes[0], scaling)
        pca_instance = PCA(n_components=components)
        features = pca_instance.fit_transform(spikes)
    elif method == "AE":
        if components == 20:
            features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                          labels=labels, gt_labels=gt_labels, index=0,
                                                          ae_type="normal",
                                                          ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                          code_size=components,
                                                          output_activation=output_activation,
                                                          loss_function=loss_function,
                                                          scale=scaling, shuff=True, nr_epochs=100)

        else:
            features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                          labels=labels, gt_labels=gt_labels, index=0,
                                                          ae_type="normal",
                                                          ae_layers=np.array([50, 40, 30, 20, 10, 5]),
                                                          code_size=components,
                                                          output_activation=output_activation,
                                                          loss_function=loss_function,
                                                          scale=scaling, shuff=True, nr_epochs=100)
    elif method == "Orthogonal AE":
        if components == 20:
            features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                          labels=labels, gt_labels=gt_labels, index=0,
                                                          ae_type="orthogonal",
                                                          ae_layers=[60, 50, 40, 30], code_size=components,
                                                          output_activation=output_activation,
                                                          loss_function=loss_function,
                                                          scale=scaling, shuff=True)
        else:
            features, labels, gt_labels = run_autoencoder(data_type="real", simulation_number=0, data=spikes,
                                                          labels=labels, gt_labels=gt_labels, index=0,
                                                          ae_type="orthogonal",
                                                          ae_layers=[40, 20, 10, 5], code_size=components,
                                                          output_activation=output_activation,
                                                          loss_function=loss_function,
                                                          scale=scaling, shuff=True)

    return compute_retard_metrics(components, scaling, features, labels, gt_labels)


def create_plot_metrics(data="C37", components=2, scaling="minmax"):
    if data=="C28":
        spikes, labels, gt_labels = read_kampff_c28()
    elif data=="C37":
        spikes, labels, gt_labels = read_kampff_c37()


    # for components in [2, 3, 20]:
    # for scaling in ["-", "minmax", "divide_amplitude"]:

    method1 = "PCA"
    # method2 = "AE"
    method2 = "AE"
    non_determenistic_runs = 10
    metrics1 = calculate_metrics_specific(spikes, labels, gt_labels, method=method1, components=components, scaling=scaling)

    sum = []
    for i in range(non_determenistic_runs):
        metrics2 = calculate_metrics_specific(spikes, labels, gt_labels, method=method2, components=components, scaling=scaling)
        sum.append(metrics2)

    sum = np.array(sum)
    metrics2 = np.mean(sum, axis=0)

    plot_metrics_clustering_eval(f"{data} - {components}D - {scaling}", metrics1[:7], metrics2[:7], method1, method2)
    plot_metrics_fe_eval(f"{data} - {components}D - {scaling}", metrics1[7:], metrics2[7:], method1, method2)



# create_plot_metrics()







# ### PCA verify
# spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
# spikes = spikes[labels != 0]
# labels = labels[labels != 0]
# compare_metrics("Full", spikes, labels, len(np.unique(labels)))
# pca_2d = PCA(n_components=20)
# pca_features = pca_2d.fit_transform(spikes)
# print(np.cumsum(np.round(pca_2d.explained_variance_ratio_, decimals=3) * 100))
# compare_metrics("PCA20D", pca_features, labels, len(np.unique(labels)))
# pca_2d = PCA(n_components=2)
# pca_features = pca_2d.fit_transform(spikes)
# compare_metrics("PCA2D", pca_features, labels, len(np.unique(labels)))
#
#
#
# ### PCA spike reconstruction
# spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
#
# spikes = spikes[labels != 0]
# labels = labels[labels != 0]
#
# chosen_spike = spikes[0]
#
# pca = PCA(n_components=2)
# spikes_pca = pca.fit_transform(spikes)
# spikes_projected = pca.inverse_transform(spikes_pca)
# loss = np.sum((spikes - spikes_projected) ** 2, axis=1).mean()
#
# plt.figure()
# plt.plot(np.arange(len(spikes[0])), spikes[0], label="original")
# # plt.plot(encoded_spike, c="red", marker="o")
# plt.plot(np.arange(len(spikes_projected[0])), spikes_projected[0], label="reconstructed")
# plt.xlabel('Time')
# plt.ylabel('Magnitude')
# plt.legend(loc="upper left")
# plt.title(f"Verify PCA 2D")
# plt.show()