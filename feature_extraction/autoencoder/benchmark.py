from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score, homogeneity_completeness_v_measure

from utils.dataset_parsing import simulations_dataset as ds
from feature_extraction.autoencoder.model_auxiliaries import get_codes
from feature_extraction.autoencoder.autoencoder import AutoencoderModel


def validate_model(autoencoder_layers, autoencoder_code_size, pt=False, noise=False):
    range_min = 1
    range_max = 96

    autoencoder = AutoencoderModel(input_size=79,
                                   encoder_layer_sizes=autoencoder_layers,
                                   decoder_layer_sizes=autoencoder_layers,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if pt == True:
        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_pt')
    elif noise == True:
        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_noise_c{autoencoder_code_size}')
    else:
        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}')

    results = []
    for simulation_number in range(range_min, range_max):
        if simulation_number == 25 or simulation_number == 44:
            continue
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

        if noise == False:
            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        pn = 25
        clustering_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

        hom, com, vm = homogeneity_completeness_v_measure(labels, clustering_labels)

        scores = [adjusted_rand_score(labels, clustering_labels),
                  adjusted_mutual_info_score(labels, clustering_labels),
                  hom,
                  com,
                  vm,
                  calinski_harabasz_score(autoencoder_features, labels),
                  davies_bouldin_score(autoencoder_features, labels),
                  silhouette_score(autoencoder_features, labels)
                  ]
        results.append(scores)

    return results
