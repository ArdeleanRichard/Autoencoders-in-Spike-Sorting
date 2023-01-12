import os

import numpy as np
from itertools import combinations, permutations
from lp_solve import lp_solve
from scipy.optimize import linprog

from validation.rank_agg1.rank_aggregator import RankAggregator
from validation.rank_agg2.rankagg import FullListRankAggregator

os.chdir("../")

METRICS = ["ARI", "AMI", "VM", "DBS", "CHS", "SS"]
METHODS = ["PCA", "ICA", "Isomap", "Shallow AE", "AE", "Tied AE", "PCA AE", "Pretrained AE", "LSTM AE", "FT AE", "WFT AE", "Orthogonal AE", "Contractive AE"]


# simulations
# for met_id, met in enumerate(METRICS):
#     print("----------------------------------------------------------------")
#     print(f"------------------------------{met}-----------------------------")
#     print("----------------------------------------------------------------")
#     ranks_list = []
#     ranks_dict = []
#     for sim in [1, 4, 16, 35]:
#         test = np.loadtxt(f"./validation/sim{sim}_analysis.csv", dtype=float, delimiter=",")
#         ranks_list.append(np.array(METHODS)[np.argsort(test[:, met_id])[::-1]].tolist())
#         ranks = {}
#         for id, value in enumerate(test[:, met_id]):
#             ranks[METHODS[id]] = value
#         ranks_dict.append(ranks)
#
#     print(ranks_list)
#     print(ranks_dict)
#     print()
#
#     ra = RankAggregator()
#     print("Instant Runoff: ", ra.instant_runoff(ranks_list))
#     print("Borda: ", ra.borda(ranks_list))
#     print("Dowdall: ", ra.dowdall(ranks_list))
#     print("Average Rank: ", ra.average_rank(ranks_list))
#     print()
#
#     FLRA = FullListRankAggregator()
#     print("Borda: ", FLRA.aggregate_ranks(ranks_dict, method='borda'))
#     print("Spearman: ", FLRA.aggregate_ranks(ranks_dict, method='spearman'))
#     print()
#     print()
#     print()


METRICS = ["DBS", "CHS", "SS"]
METHODS = ["PCA", "ICA", "Isomap", "Shallow AE", "AE", "Tied AE", "PCA AE", "Pretrained AE", "LSTM AE", "FT AE", "WFT AE", "Orthogonal AE", "Contractive AE"]


# real data
for met_id, met in enumerate(METRICS):
    print("----------------------------------------------------------------")
    print(f"------------------------------{met}-----------------------------")
    print("----------------------------------------------------------------")
    ranks_list = []
    ranks_dict = []
    for real in [4, 6, 17, 26]:
        test = np.loadtxt(f"./validation/ch{real}.csv", dtype=float, delimiter=",")
        ranks_list.append(np.array(METHODS)[np.argsort(test[:, met_id])[::-1]].tolist())
        ranks = {}
        for id, value in enumerate(test[:, met_id]):
            ranks[METHODS[id]] = value
        ranks_dict.append(ranks)

    print(ranks_list)
    print(ranks_dict)
    print()
    #
    # ra = RankAggregator()
    # print("Instant Runoff: ", ra.instant_runoff(ranks_list))
    # print("Borda: ", ra.borda(ranks_list))
    # print("Dowdall: ", ra.dowdall(ranks_list))
    # print("Average Rank: ", ra.average_rank(ranks_list))
    # print()

    FLRA = FullListRankAggregator()
    print("Borda: ", FLRA.aggregate_ranks(ranks_dict, method='borda')[1])
    # print("Spearman: ", FLRA.aggregate_ranks(ranks_dict, method='spearman'))
    # print()
    print()
    print()
