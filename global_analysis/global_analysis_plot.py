import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy.stats import stats

from utils.constants import LABEL_COLOR_MAP_SMALLER
import seaborn as sn

os.chdir("../")

def plot_box(title, data, METHODS, conditions):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    c = 'k'
    black_dict = {  # 'patch_artist': True,
        # 'boxprops': dict(color=c, facecolor=c),
        # 'capprops': dict(color=c),
        # 'flierprops': dict(color=c, markeredgecolor=c),
        'medianprops': dict(color=c),
        # 'whiskerprops': dict(color=c)
    }

    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=False, **black_dict)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'{title} for all 95 simulations',
        xlabel='Feature Extraction Method',
        ylabel='Performance',
    )

    # Now fill the boxes with desired colors
    num_boxes = len(data)

    for i in range(num_boxes):

        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        med = bp['medians'][i]

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=LABEL_COLOR_MAP_SMALLER[i % len(METHODS)]))

        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 1.1
    # bottom = 0
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(METHODS, len(conditions)), rotation=0, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    # for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
    #     fig.text(0.90, y, METHODS[id],
    #              backgroundcolor=LABEL_COLOR_MAP2[id],
    #              color='black', weight='roman', size='x-small')

    plt.savefig(f"./figures/global_plots/boxplot_{title}_global_analysis.svg")
    plt.show()







pca =               np.loadtxt(f"./global_analysis/method_evals/pca.csv", dtype=float, delimiter=",")
ica =               np.loadtxt(f"./global_analysis/method_evals/ica.csv", dtype=float, delimiter=",")
isomap =            np.loadtxt(f"./global_analysis/method_evals/isomap.csv", dtype=float, delimiter=",")
ae_shallow =        np.loadtxt(f"./global_analysis/method_evals/ae_shallow.csv", dtype=float, delimiter=",")
ae_normal =         np.loadtxt(f"./global_analysis/method_evals/ae_normal.csv", dtype=float, delimiter=",")
ae_tied =           np.loadtxt(f"./global_analysis/method_evals/ae_tied.csv", dtype=float, delimiter=",")
ae_pca =            np.loadtxt(f"./global_analysis/method_evals/ae_pca.csv", dtype=float, delimiter=",")
ae_pt =             np.loadtxt(f"./global_analysis/method_evals/ae_pt.csv", dtype=float, delimiter=",")
ae_lstm =           np.loadtxt(f"./global_analysis/method_evals/ae_lstm.csv", dtype=float, delimiter=",")
ae_fft =            np.loadtxt(f"./global_analysis/method_evals/ae_fft.csv", dtype=float, delimiter=",")
ae_wfft =           np.loadtxt(f"./global_analysis/method_evals/ae_wfft.csv", dtype=float, delimiter=",")
ae_orthogonal =     np.loadtxt(f"./global_analysis/method_evals/ae_orthogonal.csv", dtype=float, delimiter=",")
ae_contractive =    np.loadtxt(f"./global_analysis/method_evals/ae_contractive.csv", dtype=float, delimiter=",")






# T-TESTING
METHODS = ['PCA', 'ICA', 'Isomap', 'Shallow AE', 'AE', 'Tied AE', 'PCA AE', 'Pretrained AE', 'LSTM AE', 'FT AE', 'WFT AE', 'Orthogonal AE', 'Contractive AE']
metric_names = ['ARI', 'AMI', 'V-Measure', 'DBS', 'CHS', 'SS']
for metric_id, metric_name in enumerate(metric_names):
    data = []

    data.append(pca[:, metric_id].tolist())
    data.append(ica[:, metric_id].tolist())
    data.append(isomap[:, metric_id].tolist())
    data.append(ae_shallow[:, metric_id].tolist())
    data.append(ae_normal[:, metric_id].tolist())
    data.append(ae_tied[:, metric_id].tolist())
    data.append(ae_pca[:, metric_id].tolist())
    data.append(ae_pt[:, metric_id].tolist())
    data.append(ae_lstm[:, metric_id].tolist())
    data.append(ae_fft[:, metric_id].tolist())
    data.append(ae_wfft[:, metric_id].tolist())
    data.append(ae_orthogonal[:, metric_id].tolist())
    data.append(ae_contractive[:, metric_id].tolist())


    ttest_matrix = np.zeros((len(METHODS), len(METHODS)), dtype=float)
    ttest_matrix2 = np.zeros((len(METHODS), len(METHODS)), dtype=float)
    labels = np.zeros((len(METHODS), len(METHODS)), dtype=str)
    labels2 = np.zeros((len(METHODS), len(METHODS)), dtype=str)
    labels3 = np.zeros((len(METHODS), len(METHODS)), dtype=str)
    for m1_id, m1 in enumerate(METHODS):
        for m2_id, m2 in enumerate(METHODS):
            result = stats.ttest_ind(data[m1_id], data[m2_id], equal_var=True)[1] * (len(METHODS) * (len(METHODS) - 1) / 2)
            ttest_matrix2[m1_id][m2_id] = result
            if result > 0.05:
                ttest_matrix[m1_id][m2_id] = -1
                labels[m1_id][m2_id] = ""
                labels2[m1_id][m2_id] = ""
                labels3[m1_id][m2_id] = ""
            elif 0.01 < result < 0.05:
                ttest_matrix[m1_id][m2_id] = 0
                labels[m1_id][m2_id] = ""
                labels2[m1_id][m2_id] = ""
                labels3[m1_id][m2_id] = "*"
            else:
                ttest_matrix[m1_id][m2_id] = 1
                labels[m1_id][m2_id] = f"*"
                labels2[m1_id][m2_id] = f"*"
                labels3[m1_id][m2_id] = f""


    # np.savetxt(f"./figures/global/ttest_{metric_name}.csv", np.array(ttest_matrix), delimiter=",")

    df_cm = pd.DataFrame(ttest_matrix, index=METHODS, columns=METHODS)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=False, fmt="", cmap=sn.color_palette("magma", as_cmap=True)) #vmin=2, vmax=-2)
    sn.heatmap(df_cm, annot=False, annot_kws={'va': 'bottom'}, fmt="", cbar=False, cmap=sn.color_palette("magma", as_cmap=True), linewidths=5e-3, linecolor='gray', )
    #sn.heatmap(df_cm, annot=labels3, annot_kws={'va': 'center'}, fmt="", cbar=False, cmap='jet')
    #sn.heatmap(df_cm, annot=labels2, annot_kws={'va': 'top'}, fmt="", cbar=False, cmap='jet', linewidths=0.1, linecolor='black')
    plt.savefig(f'./figures/global_plots/contusion_{metric_name}_global_analysis.svg')

    plot_box(metric_name, data, METHODS, [metric_name])
