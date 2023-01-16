import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy.stats import stats

from constants import LABEL_COLOR_MAP_SMALLER
import seaborn as sn


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



# isomap = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("contractive"):
#         isomap_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         isomap.append(isomap_sim[np.argmax(isomap_sim[:, 5])])
#
# isomap = np.array(isomap)
# np.savetxt(f"./figures/global/isomap.csv", np.array(isomap), fmt="%.3f", delimiter=",")



# ae_shallow = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("shallow"):
#         ae_shallow_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_shallow.append(ae_shallow_sim[np.argmax(ae_shallow_sim[:, 5])])
#
# ae_shallow = np.array(ae_shallow)
# np.savetxt(f"./figures/global/ae_shallow.csv", np.array(ae_shallow), fmt="%.3f", delimiter=",")


# ae_normal = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("normal"):
#         ae_normal_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_normal.append(ae_normal_sim[np.argmax(ae_normal_sim[:, 5])])
#
# ae_normal = np.array(ae_normal)
# np.savetxt(f"./figures/global/ae_normal.csv", np.array(ae_normal), fmt="%.3f", delimiter=",")


# ae_tied = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("tied"):
#         ae_tied_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_tied.append(ae_tied_sim[np.argmax(ae_tied_sim[:, 5])])
#
# ae_tied = np.array(ae_tied)
# np.savetxt(f"./figures/global/ae_tied.csv", np.array(ae_tied), fmt="%.3f", delimiter=",")


# ae_pca = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("ae_pca"):
#         ae_pca_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_pca.append(ae_pca_sim[np.argmax(ae_pca_sim[:, 5])])
#
# ae_pca = np.array(ae_pca)
# np.savetxt(f"./figures/global/ae_pca.csv", np.array(ae_pca), fmt="%.3f", delimiter=",")

# ae_pt = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("ae_pt"):
#         ae_pt_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_pt.append(ae_pt_sim[np.argmax(ae_pt_sim[:, 5])])
#
# ae_pt = np.array(ae_pt)
# np.savetxt(f"./figures/global/ae_pt.csv", np.array(ae_pt), fmt="%.3f", delimiter=",")

# ae_lstm = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("lstm"):
#         ae_lstm_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_lstm.append(ae_lstm_sim[np.argmax(ae_lstm_sim[:, 5])])
#
# ae_lstm = np.array(ae_lstm)
# np.savetxt(f"./figures/global/ae_lstm.csv", np.array(ae_lstm), fmt="%.3f", delimiter=",")

# ae_fft = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("fft"):
#         ae_fft_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_fft.append(ae_fft_sim[np.argmax(ae_fft_sim[:, 5])])
#
# ae_fft = np.array(ae_fft)
# np.savetxt(f"./figures/global/ae_fft.csv", np.array(ae_fft), fmt="%.3f", delimiter=",")

# ae_wfft = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("wfft"):
#         ae_wfft_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_wfft.append(ae_wfft_sim[np.argmax(ae_wfft_sim[:, 5])])
#
# ae_wfft = np.array(ae_wfft)
# np.savetxt(f"./figures/global/ae_wfft.csv", np.array(ae_wfft), fmt="%.3f", delimiter=",")

# ae_orthogonal = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("orthogonal"):
#         ae_orthogonal_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_orthogonal.append(ae_orthogonal_sim[np.argmax(ae_orthogonal_sim[:, 5])])
#
# ae_orthogonal = np.array(ae_orthogonal)
# np.savetxt(f"./figures/global/ae_orthogonal.csv", np.array(ae_orthogonal), fmt="%.3f", delimiter=",")

# ae_contractive = []
# for file in os.listdir("./figures/global/"):
#     if file.startswith("contractive"):
#         ae_contractive_sim = np.loadtxt(f"./figures/global/{file}", dtype=float, delimiter=",")
#         ae_contractive.append(ae_contractive_sim[np.argmax(ae_contractive_sim[:, 5])])
#
# ae_contractive = np.array(ae_contractive)
# np.savetxt(f"./figures/global/ae_contractive.csv", np.array(ae_contractive), fmt="%.3f", delimiter=",")






pca =               np.loadtxt(f"./figures/global/pca.csv", dtype=float, delimiter=",")
ica =               np.loadtxt(f"./figures/global/ica.csv", dtype=float, delimiter=",")
isomap =            np.loadtxt(f"./figures/global/isomap.csv", dtype=float, delimiter=",")
ae_shallow =        np.loadtxt(f"./figures/global/ae_shallow.csv", dtype=float, delimiter=",")
ae_normal =         np.loadtxt(f"./figures/global/ae_normal.csv", dtype=float, delimiter=",")
ae_tied =           np.loadtxt(f"./figures/global/ae_tied.csv", dtype=float, delimiter=",")
ae_pca =            np.loadtxt(f"./figures/global/ae_pca.csv", dtype=float, delimiter=",")
ae_pt =             np.loadtxt(f"./figures/global/ae_pt.csv", dtype=float, delimiter=",")
ae_lstm =           np.loadtxt(f"./figures/global/ae_lstm.csv", dtype=float, delimiter=",")
ae_fft =            np.loadtxt(f"./figures/global/ae_fft.csv", dtype=float, delimiter=",")
ae_wfft =           np.loadtxt(f"./figures/global/ae_wfft.csv", dtype=float, delimiter=",")
ae_orthogonal =     np.loadtxt(f"./figures/global/ae_orthogonal.csv", dtype=float, delimiter=",")
ae_contractive =    np.loadtxt(f"./figures/global/ae_contractive.csv", dtype=float, delimiter=",")






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
