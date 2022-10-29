import keras
from keras.layers import *
from keras import Model, Sequential
from keras.regularizers import l1
import tensorflow as tf
import numpy as np
from keras import backend as K

from neural_networks.autoencoder.autoencoder_pca_principles.dense_tied_layer import DenseTied
from neural_networks.autoencoder.model_auxiliaries import get_codes, get_loss_function, get_activation_function


class AutoencoderTiedModel(Model):

    def __init__(self, input_size, encoder_layer_sizes, decoder_layer_sizes, code_size, output_activation, loss_function, dropout=0.0):
        super(AutoencoderTiedModel, self).__init__()
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


        code_layer = Dense(code_size, activation=activation, input_shape=(encoder_layer_sizes[-1],), name="code")
        encoder_layers.append(code_layer)

        decoder_layer_sizes = np.flip(decoder_layer_sizes)
        decoder_layers = []
        for index, hidden_layer_size in enumerate(decoder_layer_sizes):
            hidden_tied_layer = DenseTied(hidden_layer_size, activation='relu', tied_to=encoder_layers[len(decoder_layer_sizes)-index], use_bias=False)
            decoder_layers.append(hidden_tied_layer)

        self.autoencoder_model = Sequential()
        for layer in encoder_layers:
            self.autoencoder_model.add(layer)
            if dropout != 0:
                self.autoencoder_model.add(Dropout(0.3))
        for layer in decoder_layers:
            self.autoencoder_model.add(layer)
            if dropout != 0:
                self.autoencoder_model.add(Dropout(0.3))

        self.output_spike = Dense(self.input_size, activation=activation)
        self.autoencoder_model.add(self.output_spike)

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

        muie = self.return_autoencoder()
        lam = 100
        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)
            W = K.transpose(muie.get_layer('code').get_weights()[0])  # N_hidden x N
            h = muie.get_layer('code').output
            dh = h * (1 - h)  # N_batch x N_hidden

            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive

        def contractive_loss2(x, x_bar):
            mse = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
            W = tf.transpose(muie.get_layer('code').get_weights()[0])
            h = muie.get_layer('code').output
            dh = h * (1 - h)
            contractive = lam * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
            total_loss = mse + contractive

            return total_loss

        if self.loss_function == 'contractive':
            loss = contractive_loss2
        else:
            loss = get_loss_function(self.loss_function)



        self.autoencoder_model.compile(optimizer=opt, loss=loss)
        self.autoencoder_model.fit(training_data, training_data, epochs=epochs, verbose=verbose)




    def return_encoder(self):
        return self.encoder_model, self.autoencoder_model

    def return_autoencoder(self):
        return self.autoencoder_model
