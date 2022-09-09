import numpy as np
from keras.constraints import UnitNorm
from keras.layers import *
from keras import Model
from keras import Sequential

from feature_extraction.autoencoder.model_auxiliaries import get_activation_function, get_loss_function
from feature_extraction.autoencoder.autoencoder_pca_principles.orthogonal_weights import WeightsOrthogonalityConstraint
from feature_extraction.autoencoder.autoencoder_pca_principles.dense_tied_layer import DenseTied


class AutoencoderPCAModel(Model):

    def __init__(self, input_size, autoencoder_layers, code_size, output_activation, loss_function):
        super(AutoencoderPCAModel, self).__init__()
        self.input_size = input_size
        self.loss_function = loss_function

        self.batch_size = 16
        activation = get_activation_function(output_activation)

        encoder_layers = []
        encoder_layers.append(
            Dense(autoencoder_layers[0], activation='relu', input_shape=(input_size,), use_bias=True,
                  kernel_regularizer=WeightsOrthogonalityConstraint(autoencoder_layers[0], weightage=1., axis=0),
                  kernel_constraint=UnitNorm(axis=0))
        )
        for index, hidden_layer_size in enumerate(autoencoder_layers[1:]):
            hidden_layer = Dense(hidden_layer_size, activation='relu', input_shape=(autoencoder_layers[index],), use_bias=True,
                             kernel_regularizer=WeightsOrthogonalityConstraint(hidden_layer_size, weightage=1., axis=0),
                             kernel_constraint=UnitNorm(axis=0))
            encoder_layers.append(hidden_layer)


        code_layer = Dense(code_size, activation=activation, input_shape=(autoencoder_layers[-1],), use_bias=True,
                         kernel_regularizer=WeightsOrthogonalityConstraint(code_size, weightage=1., axis=0),
                         kernel_constraint=UnitNorm(axis=0))
        encoder_layers.append(code_layer)

        decoder_layer_sizes = np.flip(autoencoder_layers)
        decoder_layers = []
        for index, hidden_layer_size in enumerate(decoder_layer_sizes):
            hidden_tied_layer = DenseTied(hidden_layer_size, activation='relu', tied_to=encoder_layers[len(autoencoder_layers)-index], use_bias=False)
            # hidden_layer = Dense(hidden_layer_size, activation='relu')(current_layer)
            decoder_layers.append(hidden_tied_layer)

        self.autoencoder_model = Sequential()
        for layer in encoder_layers:
            self.autoencoder_model.add(layer)
        for layer in decoder_layers:
            self.autoencoder_model.add(layer)

        self.output_spike = Dense(self.input_size, activation=activation)
        self.autoencoder_model.add(self.output_spike)

        self.encoder_model = Sequential()
        for layer in encoder_layers:
            self.encoder_model.add(layer)



    def train(self, training_data, epochs=50, verbose="auto", learning_rate=0.001):
        loss = get_loss_function(self.loss_function)
        self.autoencoder_model.compile(metrics=['accuracy'],
                                       loss=loss,
                                       optimizer='sgd')
        # self.autoencoder_model.summary()

        self.autoencoder_model.fit(training_data, training_data,
                                   epochs=epochs,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   verbose=0)

    def return_encoder(self):
        return self.encoder_model, self.autoencoder_model

    def return_autoencoder(self):
        return self.autoencoder_model
