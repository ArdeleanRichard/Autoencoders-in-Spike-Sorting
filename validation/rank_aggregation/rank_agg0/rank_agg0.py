import numpy as np
from itertools import combinations, permutations
from lp_solve import lp_solve
from scipy.optimize import linprog
# A Kemeny-Young aggregation

def kendalltau_dist(rank_a, rank_b):
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) == -np.sign(rank_b[i] - rank_b[j]))
    return tau

def rankaggr_brute(ranks):
    min_dist = np.inf
    best_rank = None
    n_voters, n_candidates = ranks.shape
    for candidate_rank in permutations(range(n_candidates)):
        dist = np.sum(kendalltau_dist(candidate_rank, rank) for rank in ranks)
        if dist < min_dist:
            min_dist = dist
            best_rank = candidate_rank
    return min_dist, best_rank


def _build_graph(ranks):
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights


def rankaggr_lp(ranks):
    """Kemeny-Young optimal rank aggregation"""

    n_voters, n_candidates = ranks.shape

    # maximize c.T * x
    edge_weights = _build_graph(ranks)
    c = -1 * edge_weights.ravel()

    idx = lambda i, j: n_candidates * i + j

    # constraints for every pair
    pairwise_constraints = np.zeros(( (n_candidates * (n_candidates - 1)) // 2, n_candidates ** 2))
    for row, (i, j) in zip(pairwise_constraints, combinations(range(n_candidates), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(((n_candidates * (n_candidates - 1) * (n_candidates - 2)), n_candidates ** 2))
    for row, (i, j, k) in zip(triangle_constraints, permutations(range(n_candidates), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = np.vstack([pairwise_constraints, triangle_constraints])
    constraint_rhs = np.hstack([np.ones(len(pairwise_constraints)),
                                np.ones(len(triangle_constraints))])
    constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
                                  np.ones(len(triangle_constraints))])  # >=

    print(c.shape)
    print(constraints.shape)
    print(constraint_rhs.shape)
    print(constraint_signs.shape)
    result = linprog(c, constraints, constraint_rhs)

    x = np.array(result.x).reshape((n_candidates, n_candidates))
    aggr_rank = x.sum(axis=1)

    return aggr_rank