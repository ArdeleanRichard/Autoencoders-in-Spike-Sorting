import numpy as np
import matplotlib.pyplot as plt

def temporalize_data(X, y, lookback):
    '''
    Inputs
    X         A 2D numpy array ordered by time of shape:
              (n_observations x n_features)
    y         A 1D numpy array with indexes aligned with
              X, i.e. y[i] should correspond to X[i].
              Shape: n_observations.
    lookback  The window size to look back in the past
              records. Shape: a scalar.

    Output
    output_X  A 3D numpy array of shape:
              ((n_observations-lookback-1) x lookback x
              n_features)
    output_y  A 1D array of shape:
              (n_observations-lookback-1), aligned with X.
    '''
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)

# def temporalize_spike(X, lookback):
#     output_X = []
#     for i in range(len(X) - lookback + 1):
#         output_X.append(X[i:(i + lookback - 1)])
#     return np.squeeze(np.array(output_X))

# spike = temporalize_spikes(spike, 20, 0)
# spike = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
# spike = [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
#   [20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
#   [40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59]
#   [60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79]]
#
# spike = temporalize_spikes(spike, 20, 10)
# spike = [[ 0  1  2 ... 17 18 19]
#   [10 11 12 ... 27 28 29]
#   [20 21 22 ... 37 38 39]
#   ...
#   [40 41 42 ... 57 58 59]
#   [50 51 52 ... 67 68 69]
#   [60 61 62 ... 77 78 79]]

def temporalize_spikes(spikes, timesteps, overlap=0):
    output_X = []
    if overlap < timesteps:
        for spike in spikes:
            spike = np.pad(spike, (0, 2), 'constant')
            temp = []
            for i in range(0, len(spike), timesteps):
                if overlap != 0 and i >= overlap:
                    temp.append(spike[(i - overlap): (i + timesteps-overlap)])
                temp.append(spike[i: (i + timesteps)])
            output_X.append(temp)
    return np.squeeze(np.array(output_X))

def lstm_verify_output(training_data, lookahead, encoder, autoencoder, i=0, path=""):
    decoded_spike = autoencoder.predict(training_data[i].reshape(1, -1, lookahead))
    decoded_spike = np.array(decoded_spike)

    encoded_spike = encoder.predict(training_data[i].reshape(1, -1, lookahead))
    encoded_spike = np.array(encoded_spike)

    # decoded_spike = decoder.predict(encoded_spike)
    # decoded_spike = np.array(decoded_spike)

    plt.plot(np.arange(len(training_data[i].flatten())), training_data[i].flatten())
    # plt.plot(encoded_spike, c="red", marker="o")
    plt.plot(np.arange(len(decoded_spike[0].flatten())), decoded_spike[0].flatten())
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.title(f"Verify spike {i}")
    plt.savefig(f'{path}/spike{i}')
    # plt.show()
    plt.clf()


def lstm_verify_random_outputs(training_data, lookahead, encoder, autoencoder, verifications=0, path=""):
    random_list = np.random.choice(range(len(training_data)), verifications, replace=False)

    for random_index in random_list:
        lstm_verify_output(training_data, lookahead, encoder, autoencoder, random_index, path)


def lstm_create_code_numpy(spike, encoder, lookahead):
    return encoder.predict(spike.reshape(1,  -1, lookahead))[0]


def lstm_get_codes(training_data, encoder, lookahead):
    result = []
    for spikes in training_data:
        codes_for_spikes = lstm_create_code_numpy(spikes, encoder, lookahead)
        result.append(codes_for_spikes[1])

    return np.array(result)

# Jupyter notebook test
#
# import numpy as np
#
# def temporalize_data(X, lookback):
#     output_X = []
#     for i in range(len(X) - lookback - 1):
#         t = []
#         for j in range(1, lookback + 1):
#             t.append(X[[(i + j + 1)], :])
#         output_X.append(t)
#     return np.squeeze(np.array(output_X))
#
# def temporalize_spikes(spikes, lookback):
#     output_X = []
#     for spike in spikes:
#         temp = []
#         for i in range(0, len(spike), lookback):
#             temp.append(spike[i:(i + lookback)])
#         output_X.append(temp)
#     return np.squeeze(np.array(output_X))
#
# X = []
# for i in range(10):
#     X.append(np.arange(0, 80))
#
# X = np.array(X)
# print(X.shape)
# X = temporalize_spikes(X, 20)
# print(X.shape)
#
# print(X)
#
# def temporalize_spike(X, lookback):
#     output_X = []
#     for i in range(len(X) - lookback + 1):
#         output_X.append(X[i:(i + lookback)])
#     return np.squeeze(np.array(output_X))
#
# spike = np.arange(0, 80)
# print(spike)
# print(spike.shape)
# spike = temporalize_spike(spike, 5)
# print(spike.shape)
# print(spike)