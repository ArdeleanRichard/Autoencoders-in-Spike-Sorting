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


LABEL_COLOR_MAP = {-1: 'gray',
                   0: 'white',
                   1: 'red',
                   2: 'blue',
                   3: 'green',
                   4: 'black',
                   5: 'yellow',
                   6: 'cyan',
                   7: 'magenta',
                   8: 'tab:purple',
                   9: 'tab:orange',
                   10: 'tab:brown',
                   11: 'tab:pink',
                   12: 'lime',
                   13: 'orchid',
                   14: 'khaki',
                   15: 'lightgreen',
                   16: 'orangered',
                   17: 'salmon',
                   18: 'silver',
                   19: 'yellowgreen',
                   20: 'royalblue',
                   21: 'beige',
                   22: 'crimson',
                   23: 'indigo',
                   24: 'darkblue',
                   25: 'gold',
                   26: 'ivory',
                   27: 'lavender',
                   28: 'lightblue',
                   29: 'olive',
                   30: 'sienna',
                   31: 'salmon',
                   32: 'teal',
                   33: 'turquoise',
                   34: 'wheat',
                   }