"""Two-way matcher with ratio test.

Authors: Ayush Baid
"""
import numpy as np
import scipy.spatial.distance as distance

from frontend.matcher.matcher_base import MatcherBase


class TwoWayWithRatioTestMatcher(MatcherBase):
    """Two way matcher supporting the ratio test using OpenCV."""

    def __init__(self,
                 ratio_test_threshold: float):
        """Initializes the matcher along with various properties.

        Args:
            ratio_test_threshold (float): threshold for ratio test in [0, 1].
        """
        super().__init__()

        self.ratio_test_threshold = ratio_test_threshold

    def __generate_distance_matrix(self,
                                   vectors_1: np.ndarray,
                                   vectors_2: np.ndarray,
                                   distance_type: str
                                   ) -> np.ndarray:
        if distance_type not in ['euclidean', 'hamming']:
            raise NotImplementedError(
                'The specified distance type is not implemented')

        return distance.cdist(
            vectors_1, vectors_2, metric=distance_type)

    def __get_nearest_neighbors(self,
                                distance_matrix: np.ndarray) -> np.ndarray:

        row, _ = np.indices((distance_matrix.shape[0], 2))

        if distance_matrix.shape[1] > 2:
            top2_indices = np.argpartition(
                distance_matrix, kth=2, axis=1)[:, :2]
            top2_distances = distance_matrix[row, top2_indices]
        else:
            top2_indices = np.empty(
                (distance_matrix.shape[0], 2), dtype=np.uint32)
            top2_indices[:, 0] = 0
            top2_indices[:, 1] = 1
            top2_distances = distance_matrix

        sort_indices = np.argsort(top2_distances, axis=1)

        top2_distances = top2_distances[row, sort_indices]
        top2_indices = top2_indices[row, sort_indices]

        ratio_test_result = top2_distances[:, 0] < \
            self.ratio_test_threshold * top2_distances[:, 1]

        return top2_indices[:, 0], ratio_test_result

    def match(self,
              descriptors_im1: np.ndarray,
              descriptors_im2: np.ndarray,
              distance_type: str = 'euclidean') -> np.ndarray:
        """
        Match a pair of descriptors.

        Refer to documentation in the parent class for detailed output format.

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """

        if descriptors_im1 is None or \
                descriptors_im2 is None or \
                descriptors_im1.size == 0 or \
                descriptors_im2.size == 0:
            return np.array([], dtype=np.uint32)

        # both descriptors should have atleast two entries
        if descriptors_im1.shape[0] < 2 or descriptors_im2.shape[0] < 2:
            return np.array([], dtype=np.uint32)

        distance_matrix = self.__generate_distance_matrix(
            descriptors_im1, descriptors_im2, distance_type)

        # matching from image #1 to image #2
        nearest_indices_1to2, ratio_test_result_1to2 = \
            self.__get_nearest_neighbors(distance_matrix)

        # matching from image #2 to image #1
        nearest_indices_2to1, ratio_test_result_2to1 = \
            self.__get_nearest_neighbors(distance_matrix.T)

        # check for mutual-NN
        indices_im1 = np.arange(descriptors_im1.shape[0])
        indices_im1to2to1 = nearest_indices_2to1[nearest_indices_1to2]

        consistenty_results = indices_im1 == indices_im1to2to1

        mutual_nn_indices_im1 = np.where(consistenty_results)[0]

        match_indices = np.empty(
            (mutual_nn_indices_im1.shape[0], 2)).astype(np.uint32)
        match_indices[:, 0] = mutual_nn_indices_im1
        match_indices[:, 1] = nearest_indices_1to2[mutual_nn_indices_im1]

        ratio_test_results = np.logical_and(
            ratio_test_result_1to2[match_indices[:, 0]],
            ratio_test_result_2to1[match_indices[:, 1]]
        )

        ratio_test_valid_indices = np.where(ratio_test_results)[0]

        valid_match_indices = match_indices[ratio_test_valid_indices, :]

        valid_match_distances = distance_matrix[
            valid_match_indices[:, 0], valid_match_indices[:, 1]]

        match_confidence_ordering = np.argsort(valid_match_distances)

        return valid_match_indices[match_confidence_ordering, :]
