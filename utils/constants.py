DATA_FOLDER_PATH = "../../DATA/"
REAL_DATA_FOLDER_PATH = DATA_FOLDER_PATH + f'/TINS/'
SYNTHETIC_DATA_FOLDER_PATH = DATA_FOLDER_PATH + f'/SIMULATIONS/'

dataName = ["S1", "S2", "U", "UO", "Simulation"]
dataFiles = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]

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


LABEL_COLOR_MAP_SMALLER = {
                   0: 'lightskyblue',
                   1: 'deepskyblue',
                   2: 'violet',
                   3: 'lightcoral',
                   4: 'coral',
                   5: 'darksalmon',
                   6: 'salmon',
                   7: 'lightsalmon',
                   8: 'tomato',
                   9: 'indianred',
                   10: 'firebrick',
                   11: 'brown',
                   12: 'maroon',
                   }