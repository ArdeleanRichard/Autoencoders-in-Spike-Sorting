import keras
from keras.layers import *
from keras import Model, Sequential
from keras.regularizers import l1
import tensorflow as tf
import numpy as np

from feature_extraction.autoencoder.autoencoder_pca_principles.dense_tied_layer import DenseTied
from feature_extraction.autoencoder.autoencoder_pca_principles.dense_tied_layer2 import DenseTranspose
from feature_extraction.autoencoder.model_auxiliaries import get_codes, get_loss_function, get_activation_function
from feature_extraction.autoencoder.softplusplus_activation import softplusplus


class AutoencoderTied2Model(Model):

    def __init__(self, input_size, encoder_layer_sizes, decoder_layer_sizes, code_size, output_activation, loss_function):
        super(AutoencoderTied2Model, self).__init__()
        self.input_size = input_size
        self.loss_function = loss_function
        activation = get_activation_function(output_activation)


        encoder_layers = []
        encoder_layers.append(
            Dense(encoder_layer_sizes[0], activation='relu', input_shape=(input_size,))
        )
        for index, hidden_layer_size in enumerate(encoder_layer_sizes[1:]):
            hidden_layer = Dense(hidden_layer_size, activation='relu', input_shape=(encoder_layer_sizes[index], ))
            encoder_layers.append(hidden_layer)


        code_layer = Dense(code_size, activation=activation, input_shape=(encoder_layer_sizes[-1],))
        encoder_layers.append(code_layer)

        decoder_layer_sizes = np.flip(decoder_layer_sizes)
        decoder_layers = []
        for index, hidden_layer_size in enumerate(decoder_layer_sizes):
            hidden_tied_layer = DenseTranspose(encoder_layers[len(decoder_layer_sizes)-index], activation='relu')
            decoder_layers.append(hidden_tied_layer)

        self.outputs = Reshape([2, 5127, -1])

        self.autoencoder_model = Sequential()
        for layer in encoder_layers:
            self.autoencoder_model.add(layer)
        for layer in decoder_layers:
            self.autoencoder_model.add(layer)
        self.autoencoder_model.add(self.outputs)

        self.encoder_model = Sequential()
        for layer in encoder_layers:
            self.encoder_model.add(layer)



    def pre_train(self, training_data, autoencoder_layer_sizes, epochs):
        encoder_layer_weights = []
        decoder_layer_weights = []

        current_training_data = training_data
        current_input_size = self.input_size

        for code_size in autoencoder_layer_sizes:
            input = Input(shape=(current_input_size,))
            code_out = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-8))(input)
            code_in = Input(shape=(code_size,))
            output = Dense(current_input_size, activation='tanh')(code_in)

            encoder = Model(input, code_out)
            decoder = Model(code_in, output)

            input = Input(shape=(current_input_size,))
            code = encoder(input)
            decoded = decoder(code)

            pretrained = Model(input, decoded)


            # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
            pretrained.compile(optimizer='adam', loss=self.loss_function)
            pretrained.fit(current_training_data, current_training_data, epochs=epochs)

            encoder_layer_weights.append(pretrained.get_weights()[0])
            encoder_layer_weights.append(pretrained.get_weights()[1])
            decoder_layer_weights.append(pretrained.get_weights()[3])
            decoder_layer_weights.append(pretrained.get_weights()[2])

            current_input_size = code_size
            current_training_data = get_codes(current_training_data, encoder)

        decoder_layer_weights = decoder_layer_weights[::-1]
        encoder_layer_weights.extend(decoder_layer_weights)
        return encoder_layer_weights

    def train(self, training_data, epochs=50, verbose="auto", learning_rate=0.001):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = get_loss_function(self.loss_function)
        self.autoencoder_model.compile(optimizer=opt, loss=loss)
        self.autoencoder_model.fit(training_data, training_data, epochs=epochs, verbose=verbose)


    def return_encoder(self):
        return self.encoder_model, self.autoencoder_model

    def return_autoencoder(self):
        return self.autoencoder_model
