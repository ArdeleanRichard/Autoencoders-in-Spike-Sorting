import os

from dataset_parsing import simulations_dataset as ds
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from validation.performance import compute_metrics_by_kmeans

os.chdir("../../")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

SIM_NR = 4
spikes, labels = ds.get_dataset_simulation_pca_2d(simNr=SIM_NR, align_to_peak=2)

RUNS = 1000

METHODS = ['K-Means']
metric_names = ['ARI', 'AMI', 'V-Measure', 'FMI', 'DBS', 'CHS', 'SS']
box_colors = ['darkkhaki', 'royalblue', 'pink', 'lightgreen']
weights = ['bold', 'semibold', 'light', 'heavy']


def plot_box(data):
    fig, ax1 = plt.subplots(figsize=(10, 6),)
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
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'K-Means Variability in {RUNS} executions',
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

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % len(METHODS)]))

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

    plt.savefig(f"./validation/variability_kmeans/kmeans_variability_{RUNS}_boxplot")
    plt.show()


def main(program):
    if program == 'save':
        all_metrics = []
        for i in range(RUNS):
            if i % 10 == 0:
                print(i)
            metrics = compute_metrics_by_kmeans(spikes, labels)
            all_metrics.append(metrics)
        all_metrics = np.array(all_metrics)

        np.savetxt(f"./validation/variability_kmeans/kmeans_variability{RUNS}.csv", all_metrics, fmt="%.2f", delimiter=",")
    elif program == 'plot':
        all_metrics = np.loadtxt(f"./validation/variability_kmeans/kmeans_variability{RUNS}.csv", dtype=float, delimiter=",")
        print(all_metrics.shape)
        all_metrics[:, 5] = all_metrics[:, 5] / np.amax(all_metrics[:, 5])
        all_metrics = all_metrics.T.tolist()

        plot_box(all_metrics)


# main("save")
main("plot")