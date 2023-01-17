import os

import numpy as np
from itertools import combinations, permutations
from lp_solve import lp_solve
from scipy.optimize import linprog

from validation.rank_aggregation.rank_agg1.rank_aggregator import RankAggregator
from validation.rank_aggregation.rank_agg2.rankagg import FullListRankAggregator

os.chdir("../")

METRICS = ["ARI", "AMI", "VM", "DBS", "CHS", "SS"]
METHODS = ["PCA", "ICA", "Isomap", "Shallow AE", "AE", "Tied AE", "PCA AE", "Pretrained AE", "LSTM AE", "FT AE", "WFT AE", "Orthogonal AE", "Contractive AE"]



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




for met_id, met in enumerate(METRICS):
    print("----------------------------------------------------------------")
    print(f"------------------------------{met}-----------------------------")
    print("----------------------------------------------------------------")

    ranks_list = []
    ranks_dict = []
    for id in range(len(pca)):
        method_results = np.array([pca[id][met_id],
                          ica[id][met_id],
                          isomap[id][met_id],
                          ae_shallow[id][met_id],
                          ae_normal[id][met_id],
                          ae_tied[id][met_id],
                          ae_pca[id][met_id],
                          ae_pt[id][met_id],
                          ae_lstm[id][met_id],
                          ae_fft[id][met_id],
                          ae_wfft[id][met_id],
                          ae_orthogonal[id][met_id],
                          ae_contractive[id][met_id],
                          ])
        ranks_list.append(np.array(METHODS)[np.argsort(method_results)[::-1]].tolist())
        # print(ranks_list)

        ranks = {}
        for id, value in enumerate(method_results):
            ranks[METHODS[id]] = value
        ranks_dict.append(ranks)

    FLRA = FullListRankAggregator()
    print("Borda: ", FLRA.aggregate_ranks(ranks_dict, method='borda')[1])
    # print("Spearman: ", FLRA.aggregate_ranks(ranks_dict, method='spearman'))
    print()
    print()
    print()
