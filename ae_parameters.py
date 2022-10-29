PLOT_PATH = './figures/'
MODEL_PATH = './weights/'

simulation_number = 1
output_activation = 'tanh'
# output_activation = 'linear'
# output_activation = 'spp'

loss_function = 'mse'
# loss_function = 'bce'
# loss_function = 'ce'
# loss_function = 'cce'
# loss_function = 'scce'

# DECODER order is reversed
# autoencoder_layer_sizes = [100,90,80,70,60]
autoencoder_layer_sizes = [70,60,50,40,30]
autoencoder_selected_layer_sizes = [40,35,30,25]
autoencoder_expanded_layer_sizes = [70,80,90,100,110]
autoencoder_cascade_layer_sizes = [90,80,70,60,50,40,30]
autoencoder_single_sim_layer_sizes = [70,60,50,40,30,20,10,5]
autoencoder_single_sim_code_size = 2
# autoencoder_code_size = 50
autoencoder_code_size = 20
autoencoder_expanded_code_size = 120
lstm_layer_sizes = [64, 32]
lstm_code_size = 20
lstm_single_sim_layer_sizes = [64, 32,16,8,4]
lstm_single_sim_code_size = 2
