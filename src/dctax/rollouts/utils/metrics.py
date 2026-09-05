"""Distances between the answer distributions a sweep produces.

    from dctax.rollouts.utils import tvd_matrix

    matrix = tvd_matrix(arms, vocabulary)   # matrix[i, j] is TVD between boundaries i and j

Separate from `load` because these take arms that are already in hand rather than reading
anything, and both the convergence and the counterfactual analyses need them.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist, squareform


def tvd_matrix(arms: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    """Pairwise total variation distance between every boundary's answer distribution.

        TVD(P_i, P_j) = 0.5 * sum_a |P_i(a) - P_j(a)|

    Each boundary becomes a row of probabilities over the trace's vocabulary, so the L1
    distance between two rows is twice their TVD. `pdist` does the whole triangle in C, which
    matters because a 950-boundary GLM trace has 450k pairs and a Python loop over them costs
    minutes per trace.

    TVD ignores entities neither side used, so unlike a smoothed KL it does not move with the
    size of the vocabulary.
    """
    index = {entity: i for i, entity in enumerate(vocabulary)}
    table = np.zeros((len(arms), len(vocabulary)), dtype=np.float64)
    for row, arm in enumerate(arms):
        for entity, count in Counter(arm).items():
            table[row, index[entity]] = count
        table[row] /= len(arm)
    return squareform(pdist(table, metric="cityblock")) / 2.0


def forward_curves(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    """Each boundary summarised over its comparisons with every later boundary.

    The worst case is what a convergence rule would threshold; the median says how variable
    the disagreement is at that depth. The final boundary has nothing after it and is not
    reported, so both curves are one shorter than the matrix.
    """
    maxima, medians = [], []
    for i in range(len(matrix) - 1):
        forward = matrix[i, i + 1:]
        maxima.append(float(forward.max()))
        medians.append(float(np.median(forward)))
    return maxima, medians
