import keras
from keras.layers import *
from keras import Model
from keras.regularizers import l1
import tensorflow as tf
import numpy as np
from keras import backend as K

from neural_networks.autoencoder.model_auxiliaries import get_codes, get_loss_function, get_activation_function


class AutoencoderModel(Model):
    """
    Autoencoder Model - automatic creation of the model
    """
    def __init__(self, input_size, encoder_layer_sizes, decoder_layer_sizes, code_size, output_activation, loss_function, dropout=0.0):
        super(AutoencoderModel, self).__init__()
        self.input_size = input_size
        self.loss_function = loss_function
        activation = get_activation_function(output_activation)

        self.input_spike = Input(shape=(self.input_size,))

        current_layer = self.input_spike
        for hidden_layer_size in encoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu')(current_layer)
            if dropout == True:
                dropout_layer = Dropout(dropout)(hidden_layer)
                current_layer = dropout_layer
            else:
                current_layer = hidden_layer

        self.code_result = Dense(code_size, activation=activation, activity_regularizer=l1(10e-7), name="code")(current_layer)

        decoder_layer_sizes = np.flip(decoder_layer_sizes)

        # self.code_input = Input(shape=(code_size,))
        current_layer = self.code_result
        for hidden_layer_size in decoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu')(current_layer)
            if dropout != 0:
                dropout_layer = Dropout(dropout)(hidden_layer)
                current_layer = dropout_layer
            else:
                current_layer = hidden_layer

        self.output_spike = Dense(self.input_size, activation=activation)(current_layer)

        self.autoencoder = Model(self.input_spike, self.output_spike)

        self.encoder = Model(self.input_spike, self.code_result)


    def pre_train(self, training_data, autoencoder_layer_sizes, epochs):
        """
        Greedy layer-wise pretraining [1]
        [1] A. Sagheer and M. Kotb, ‘Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems’, Sci Rep, vol. 9, no. 1, p. 19038, Dec. 2019, doi: 10.1038/s41598-019-55320-6.
        :param training_data: matrix - the points of the dataset
        :param autoencoder_layer_sizes: vector - the number of neurons per layer
        :param epochs: integer - the number of epochs for pretraining
        """
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
            pretrained.fit(current_training_data, current_training_data, epochs=epochs, verbose=0)

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
        """
        Autoencoder training for a number of epochs
        :param training_data: matrix - the points of the dataset
        :param epochs: integer - the number of epochs for pretraining
        :param learning_rate: integer - the learning rate
        """

        model_var = self.return_autoencoder()
        lam = 100
        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)
            W = K.transpose(model_var.get_layer('code').get_weights()[0])  # N_hidden x N
            h = model_var.get_layer('code').output
            dh = h * (1 - h)  # N_batch x N_hidden

            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive

        def contractive_loss2(x, x_bar):
            mse = tf.reduce_mean(tf.keras.losses.mse(x, x_bar))
            W = tf.transpose(model_var.get_layer('code').get_weights()[0])
            h = model_var.get_layer('code').output
            dh = h * (1 - h)
            contractive = lam * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
            total_loss = mse + contractive

            return total_loss

        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        if self.loss_function == 'contractive':
            loss = contractive_loss
        else:
            loss = get_loss_function(self.loss_function)

        self.autoencoder.compile(optimizer=opt, loss=loss)
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose)


    def return_encoder(self):
        return self.encoder, self.autoencoder

    def return_autoencoder(self):
        return self.autoencoder
