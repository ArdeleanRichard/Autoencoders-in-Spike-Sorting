from keras.layers import *
from keras import Model
from keras.regularizers import l1

import numpy as np


# model.add(RepeatVector(3))
# repeats input n times
# now: model.output_shape == (x, y, 3)
# TimeDistributed(layer)
# This wrapper allows to apply a layer to every temporal slice of an input.
class LSTMAutoencoderModel(Model):

    def __init__(self, input_size, lstm_layer_sizes, code_size):
        super(LSTMAutoencoderModel, self).__init__()

        self.input_size = input_size

        self.input_spike = Input(shape=(self.input_size[1], self.input_size[2]))

        current_layer = self.input_spike
        for hidden_layer_size in lstm_layer_sizes:
            hidden_layer_lstm = LSTM(hidden_layer_size, activation='relu', return_sequences=True)(current_layer)
            current_layer = hidden_layer_lstm

        # self.code_result = LSTM(code_size, activation='tanh', return_sequences=True, activity_regularizer=l1(10e-7))(current_layer)
        self.code_result = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-7))(current_layer)

        lstm_layer_sizes = np.flip(lstm_layer_sizes)

        # self.code_input = Input(shape=(code_size,))
        current_layer = self.code_result
        for hidden_layer_size in lstm_layer_sizes:
            hidden_layer_lstm = LSTM(hidden_layer_size, activation='relu', return_sequences=True)(current_layer)
            current_layer = hidden_layer_lstm

        # self.output_spike = TimeDistributed(Dense(self.input_size[1], activation='tanh'))(current_layer)
        # assertion error
        self.output_spike = TimeDistributed(Dense(self.input_size[2], activation='tanh'))(current_layer)

        self.autoencoder = Model(self.input_spike, self.output_spike)

        self.encoder = Model(self.input_spike, self.code_result)

    def train(self, training_data, epochs=50, verbose="auto"):
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose)

    def return_encoder(self):
        return self.encoder, self.autoencoder
