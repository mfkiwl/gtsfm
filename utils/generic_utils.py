"""
General purpose utilities common to GTSFM

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np


def find_closest_match_greedy(distance_matrix: np.ndarray,
                              dist_threshold: float) -> Tuple[np.ndarray, np.array]:
    """
    Find the closest matches using greedy algorithm and satifying the 1:1 constraint.

    Constraints followed:
    1. A match between i^th entry in set #1 and j^th entry in set #2 is
       possible only when i = argmin(d(k,j)) and j = argmin(d(i,k)).
    TODO(ayush): confirm

    Args:
        distance_matrix (np.ndarray): matrix with every pairwise distance
                                      between two sets
        dist_threshold (float): maximum distance allowed for a valid match

    Returns:
        Tuple[np.ndarray, np.array]: index of the matches, and corresponding
                                     distances
    """

    if distance_matrix is None or distance_matrix.size == 0:
        return np.array([]), np.array([])

    is_available_1 = np.full((np.shape(distance_matrix)[0],), True)
    is_available_2 = np.full((np.shape(distance_matrix)[1],), True)

    sorted_indices = np.transpose(np.vstack(np.unravel_index(
        np.argsort(np.ravel(distance_matrix)),
        distance_matrix.shape
    )))

    match_idxes = []
    match_distances = []

    for row_idx in range(sorted_indices.shape[0]):
        idx1 = sorted_indices[row_idx, 0]
        idx2 = sorted_indices[row_idx, 1]

        if dist_threshold is not None and distance_matrix[idx1, idx2] > dist_threshold:
            break

        if is_available_1[idx1] and is_available_2[idx2]:
            is_available_1[idx1] = False
            is_available_2[idx2] = False

            match_idxes.append([idx1, idx2])
            match_distances.append(distance_matrix[idx1, idx2])

    return np.array(match_idxes), np.array(match_distances)
