import os

import numpy as np

from ae_function import run_autoencoder
from validation.performance import compute_metrics_by_kmeans
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

os.chdir("../../")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300



RUNS = 100
TESTS = 1

METHODS = ['Autoencoder']
metric_names = ['ARI', 'AMI', 'V-Measure', 'DBS', 'CHS', 'SS']
box_colors = ['darkkhaki', 'royalblue', 'pink', 'lightgreen']
weights = ['bold', 'semibold', 'light', 'heavy']



def plot_box(data, text="", other_methods=[]):
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

        if other_methods != []:
            ax1.plot(np.average(med.get_xdata()), np.average(other_methods[0][i]), color='r', marker='X', markeredgecolor='k')
            ax1.plot(np.average(med.get_xdata()), np.average(other_methods[1][i]), color='g', marker='D', markeredgecolor='k')
            ax1.plot(np.average(med.get_xdata()), np.average(other_methods[2][i]), color='b', marker='P', markeredgecolor='k')

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

    plt.savefig(f"./validation/variability_ae/aept_sim4_variability_{RUNS}_boxplot")
    # plt.savefig(f"./validation_ae/ae_normal_depth{depth}_sim4_variability_{RUNS}_HU_boxplot")
    plt.show()




RUNS = 100

EPOCHS = 50
LAYERS = [70, 60, 50, 40, 30, 20, 10, 5]
LAYER_CONFIGS = [
    [40],
    [40, 20],
    [40, 20, 10],
    [40, 20, 10, 5],
    [60, 40, 20, 10, 5],
    [60, 40, 30, 20, 10, 5],
    [70, 60, 50, 40, 30, 20, 10, 5],
    [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
]


def main(program):
    if program == 'save':
        ae_type = 'normal'
        # for SIM_NR in [1,4,16,35]:
        for SIM_NR in [4]:
            print(f"SIM{SIM_NR}")
            for LAYERS in LAYER_CONFIGS:
                metrics = []
                for i in range(1, RUNS):
                    print(i)
                    features, gt, _ = run_autoencoder(data_type="sim", simulation_number=SIM_NR,
                                                      data=None, labels=None, gt_labels=None, index=None,
                                                      ae_type=ae_type, ae_layers=np.array(LAYERS), code_size=2,
                                                      output_activation='tanh', loss_function='mse', scale="minmax",
                                                      nr_epochs=EPOCHS, dropout=0.2, weight_init='he_uniform',
                                                      doPlot=False, verbose=0, shuff=True)
                    # scatter_plot.plot(f'Autoencoder on Sim{SIM_NR}', features, gt, marker='o')
                    # plt.savefig(f"./feature_extraction/autoencoder/analysis/" + f'{ae_type}_sim{SIM_NR}_plot{i}')
                    met = compute_metrics_by_kmeans(features, gt)
                    metrics.append(met)
                # np.savetxt(f"./validation_ae/ae_{ae_type}_sim{SIM_NR}_variability_{RUNS}.csv", np.around(a=np.array(metrics).transpose(), decimals=3), delimiter=",")
                # np.savetxt(f"./validation_ae/ae_{ae_type}_depth{len(LAYERS)}_sim{SIM_NR}_variability_{RUNS}.csv", metrics, fmt="%.3f", delimiter=",")
                np.savetxt(
                    f"./validation/variability_ae/ae_{ae_type}_depth{len(LAYERS)}_sim{SIM_NR}_variability_{RUNS}_HU_do20.csv",
                    metrics, fmt="%.3f", delimiter=",")

    elif program == 'plot':
        # for depth in [1,2,3,4,5,6,8,14]:
        #     all_metrics = np.loadtxt(f"./validation_ae/ae_normal_depth{depth}_sim4_variability_{RUNS}_HU.csv", dtype=float, delimiter=",")
        #     # print(all_metrics.shape)
        #     all_metrics[:, 5] = all_metrics[:, 5] / np.amax(all_metrics[:, 5])
        #     all_metrics = all_metrics[~np.all(all_metrics == 0, axis=1)]
        #     all_metrics = all_metrics.T.tolist()
        #     plot_box(all_metrics, f"Depth {depth} ")

        SIM_NR = 1
        all_metrics = np.loadtxt(f"./validation/variability_ae/ae_pytorch_sim{SIM_NR}_variability_{RUNS}_init_mod4.csv", dtype=float, delimiter=",")
        # print(all_metrics.shape)

        sim1_pca_scores = np.array([0.5, 0.74, 0.74, 1, 12020.59, 0.26])
        sim1_ica_scores = np.array([0.48, 0.72, 0.73, 1.05, 9929.23, 0.24])
        # sim1_isomap_scores = np.array([0.55, 0.79, 0.79, 1.33, 33800.12, 0.31])
        sim1_isomap_scores = np.array([0.62, 0.83, 0.83, 0.61, 59532.32, 0.51])

        all_metrics = np.delete(all_metrics, 4, 1)  # delete fmi, not used in article
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
        plot_box(all_metrics, f"", other_methods=[sim1_pca_scores, sim1_ica_scores, sim1_isomap_scores])

main('plot')