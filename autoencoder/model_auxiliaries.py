import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from feature_extraction.autoencoder.softplusplus_activation import softplusplus


def get_activation_function(activation_function):
    if activation_function == 'spp':
        return softplusplus
    else:
        return activation_function


def get_loss_function(loss_function):
    if loss_function == 'mse':
        return loss_function
    elif loss_function == 'cce':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss_function == 'scce':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss_function == 'bce':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_function == 'ce':
        return crossentropy

def crossentropy(actual, predicted):
    return -np.sum(actual * np.log2(predicted))



def verify_output(training_data, encoder, autoencoder, i=0, path=""):
    decoded_spike = autoencoder.predict(training_data[i].reshape(1, -1))
    decoded_spike = np.array(decoded_spike)

    encoded_spike = encoder.predict(training_data[i].reshape(1, -1))
    encoded_spike = np.array(encoded_spike)

    # decoded_spike = decoder.predict(encoded_spike)
    # decoded_spike = np.array(decoded_spike)

    plt.plot(np.arange(len(training_data[i])), training_data[i], label="original")
    # plt.plot(encoded_spike, c="red", marker="o")
    plt.plot(np.arange(len(decoded_spike[0])), decoded_spike[0], label="reconstructed")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike {i}")
    plt.savefig(f'{path}/spike{i}')
    plt.clf()
    plt.cla()
    # plt.show()

def verify_output(training_data, encoder, autoencoder, i=0, path="", plot_name=""):
    decoded_spike = autoencoder.predict(training_data[i].reshape(1, -1))
    decoded_spike = np.array(decoded_spike)

    encoded_spike = encoder.predict(training_data[i].reshape(1, -1))
    encoded_spike = np.array(encoded_spike)

    # decoded_spike = decoder.predict(encoded_spike)
    # decoded_spike = np.array(decoded_spike)

    plt.figure()
    plt.plot(np.arange(len(training_data[i])), training_data[i], label="original")
    # plt.plot(encoded_spike, c="red", marker="o")
    plt.plot(np.arange(len(decoded_spike[0])), decoded_spike[0], label="reconstructed")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike {i}")
    plt.savefig(f'{path}/{plot_name}')
    plt.clf()
    plt.cla()
    plt.close()
    # plt.show()

def verify_output_one(chosen_spike, encoder, autoencoder, path="", plot_name=""):
    decoded_spike = autoencoder.predict(chosen_spike.reshape(1, -1))
    decoded_spike = np.array(decoded_spike)

    encoded_spike = encoder.predict(chosen_spike.reshape(1, -1))
    encoded_spike = np.array(encoded_spike)

    # decoded_spike = decoder.predict(encoded_spike)
    # decoded_spike = np.array(decoded_spike)

    plt.figure()
    plt.plot(np.arange(len(chosen_spike)), chosen_spike, label="original")
    # plt.plot(encoded_spike, c="red", marker="o")
    plt.plot(np.arange(len(decoded_spike[0])), decoded_spike[0], label="reconstructed")
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend(loc="upper left")
    plt.title(f"Verify spike")
    plt.savefig(f'{path}/{plot_name}')
    plt.clf()
    plt.cla()
    # plt.show()

def verify_random_outputs(training_data, encoder, autoencoder, verifications=0, path=""):
    random_list = np.random.choice(range(len(training_data)), verifications, replace=False)

    for random_index in random_list:
        verify_output(training_data, encoder, autoencoder, random_index, path)


def create_code_numpy(spike, encoder):
    return encoder.predict(spike.reshape(1, -1))[0]


def get_codes(training_data, encoder):
    return np.apply_along_axis(create_code_numpy, 1, training_data, encoder)


def create_reconstruction_numpy(spike, autoencoder):
    return autoencoder.predict(spike.reshape(1, -1))[0]


def get_reconstructions(training_data, autoencoder):
    return np.apply_along_axis(create_reconstruction_numpy, 1, training_data, autoencoder)


def create_plot_folder(file_folder, inner_folder=""):
    """
    Create folder for file (file name = folder name) in figures folder
    And create an inner folder in the aforementioned folder
    :param fcs_file_name:
    :param inner_folder:

    :return None
    """
    # create folder for figures
    cwd = os.getcwd()
    path = os.path.join(cwd, "figures", file_folder)
    if not os.path.exists(path):
        os.mkdir(path)

    # create folder for figures
    cwd = os.getcwd()
    path = os.path.join(cwd, "figures", file_folder, f"{inner_folder}")
    if not os.path.exists(path):
        os.mkdir(path)

