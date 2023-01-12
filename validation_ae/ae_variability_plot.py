import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import os

os.chdir("../")


RUNS = 100
TESTS = 1

METHODS = ['Autoencoder']
metric_names = ['ARI', 'AMI', 'V-Measure', 'DBS', 'CHS', 'SS']
box_colors = ['darkkhaki', 'royalblue', 'pink', 'lightgreen']
weights = ['bold', 'semibold', 'light', 'heavy']


def plot_box(data, text=""):
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
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',  alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'Autoencoder {text}Variability in {RUNS} executions',
        xlabel='Metrics',
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
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % len(METHODS)]))

        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k')

        ax1.plot(np.average(med.get_xdata()), np.average(sim1_pca_scores[i]), color='r', marker='X', markeredgecolor='k')
        ax1.plot(np.average(med.get_xdata()), np.average(sim1_ica_scores[i]), color='g', marker='D', markeredgecolor='k')
        ax1.plot(np.average(med.get_xdata()), np.average(sim1_isomap_scores[i]), color='b', marker='P', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.1
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(metric_names, len(METHODS)), rotation=0, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
        fig.text(0.90, y, METHODS[id],
                 backgroundcolor=box_colors[id],
                 color='black', weight='roman', size='x-small')

    fig.text(0.90, 0.05,   "PCA",       backgroundcolor="r", color='black', weight='roman', size='x-small')
    fig.text(0.90, 0.09,   "ICA",       backgroundcolor="g", color='black', weight='roman', size='x-small')
    fig.text(0.90, 0.13,   "Isomap",    backgroundcolor="b", color='black', weight='roman', size='x-small')

    plt.savefig(f"./validation_ae/aept_sim4_variability_{RUNS}_boxplot")
    # plt.savefig(f"./validation_ae/ae_normal_depth{depth}_sim4_variability_{RUNS}_HU_boxplot")
    plt.show()


# for depth in [1,2,3,4,5,6,8,14]:
#     all_metrics = np.loadtxt(f"./validation_ae/ae_normal_depth{depth}_sim4_variability_{RUNS}_HU.csv", dtype=float, delimiter=",")
#     # print(all_metrics.shape)
#     all_metrics[:, 5] = all_metrics[:, 5] / np.amax(all_metrics[:, 5])
#     all_metrics = all_metrics[~np.all(all_metrics == 0, axis=1)]
#     all_metrics = all_metrics.T.tolist()
#     plot_box(all_metrics, f"Depth {depth} ")

SIM_NR = 1
all_metrics = np.loadtxt(f"./validation_ae/ae_pytorch_sim{SIM_NR}_variability_{RUNS}_init_mod4.csv", dtype=float, delimiter=",")
# print(all_metrics.shape)

sim1_pca_scores =    np.array([0.5, 0.74, 0.74, 1, 12020.59, 0.26])
sim1_ica_scores =    np.array([0.48, 0.72, 0.73, 1.05, 9929.23, 0.24])
# sim1_isomap_scores = np.array([0.55, 0.79, 0.79, 1.33, 33800.12, 0.31])
sim1_isomap_scores = np.array([0.62, 0.83, 0.83, 0.61, 59532.32, 0.51])

all_metrics = np.delete(all_metrics, 4, 1) # delete fmi, not used in article
# all_metrics[:, 4] = all_metrics[:, 4] / np.amax(all_metrics[:, 4])

# ----------
chs_max = max([np.amax(all_metrics[:, 4]), sim1_pca_scores[4], sim1_ica_scores[4], sim1_isomap_scores[4]])
dbs_max = max([np.amax(all_metrics[:, 3]), sim1_pca_scores[3], sim1_ica_scores[3], sim1_isomap_scores[3]])

all_metrics[:, 4] = all_metrics[:, 4] / chs_max
sim1_pca_scores[4] = sim1_pca_scores[4] / chs_max
sim1_ica_scores[4] = sim1_ica_scores[4] / chs_max
sim1_isomap_scores[4] = sim1_isomap_scores[4] / chs_max

all_metrics[:, 3] = all_metrics[:, 3] / dbs_max
sim1_pca_scores[3] = sim1_pca_scores[3] / dbs_max
sim1_ica_scores[3] = sim1_ica_scores[3] / dbs_max
sim1_isomap_scores[3] = sim1_isomap_scores[3] / dbs_max

# -------

all_metrics = all_metrics[~np.all(all_metrics == 0, axis=1)]
all_metrics = all_metrics.T.tolist()
print(all_metrics)
plot_box(all_metrics, f"")