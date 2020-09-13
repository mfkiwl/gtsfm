"""Generic Matcher based upon OpenCV's APIs. 

Optional features:
1. Ratio test
2. Mutual nearest neighbor
3. One-to-one constraint

Authors: Ayush Baid
"""
import cv2 as cv
import numpy as np

from frontend.matcher.matcher_base import MatcherBase


class GenericMatcher(MatcherBase):
    """Descriptor matcher using OpenCV."""

    def __init__(self,
                 ratio_test_threshold: float = None,
                 is_mutual: bool = True,
                 is_bijection: bool = False):
        """Initializes the matcher along with various properties.

        Args:
            ratio_test_threshold (float, optional): threshold for the ratio test
                                                    in [0, 1]. Defaults to None.
            is_mutual (bool, optional): flag to enforce mutual NN (two-way)
                                        match. Defaults to True.
            is_bijection (bool, optional): flag to enforce bijection in the
                                           result, i.e. each feature is
                                           selected atmost once. Defaults
                                           to False.

        Raises:
            NotImplementedError: incompatible configurations
        """
        super().__init__()

        self.ratio_test_threshold = ratio_test_threshold

        if is_mutual and not is_bijection:
            raise NotImplementedError(
                'Mutual-NN automatically enforces a bijection')

        self.is_mutual = is_mutual
        self.is_bijection = is_bijection

        self.__matcher_function = self.__match_without_ratio_test \
            if self.ratio_test_threshold is None else \
            self.__match_with_ratio_test

    def __enforce_bijection(self, match_indices: np.ndarray) -> np.ndarray:
        """Enforce a bijection on matching results by discarding matches which
        violates the 1:1 constraint.

        Args:
            match_indices (np.ndarray): Nx2 matrix where each row is a match,
                                        sorted in descending order of
                                        confidence.

        Returns:
            np.ndarray: subset of match_indices which have a 1:1 constraint.
        """
        consumed_indices_im1 = set()
        consumed_indices_im2 = set()

        results = []
        for row in match_indices:
            if row[0] not in consumed_indices_im1 and row[1] not in consumed_indices_im2:
                results.append(row)
                consumed_indices_im1.add(row[0])
                consumed_indices_im2.add(row[1])

        return np.array(results).astype(np.uint32)

    def __match_without_ratio_test(self,
                                   descriptors_im1: np.ndarray,
                                   descriptors_im2: np.ndarray,
                                   distance_type: str = 'euclidean'
                                   ) -> np.ndarray:
        """Match descriptors (all valid) without the ratio test

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """
        if distance_type == 'euclidean':
            distance_metric = cv.NORM_L2
            descriptors_dtype = np.float32
        elif distance_type == 'hamming':
            distance_metric = cv.NORM_HAMMING
            descriptors_dtype = np.uint8
        else:
            raise NotImplementedError(
                'The specified distance type is not implemented')

        opencv_obj = cv.BFMatcher(
            normType=distance_metric, crossCheck=self.is_mutual)

        matches = opencv_obj.match(
            descriptors_im1.astype(descriptors_dtype),
            descriptors_im2.astype(descriptors_dtype))

        matches.sort(key=lambda r: r.distance)

        return np.array(
            [[m.queryIdx, m.trainIdx] for m in matches]
        ).astype(np.uint32)

    def __match_with_ratio_test(self,
                                descriptors_im1: np.ndarray,
                                descriptors_im2: np.ndarray,
                                distance_type: str = 'euclidean') -> np.ndarray:
        """Match descriptors (all valid) with ratio test.

        Args:
            descriptors_im1 (np.ndarray): descriptors from image #1
            descriptors_im2 (np.ndarray): descriptors from image #2
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: match indices (sorted by confidence)
        """
        if distance_type == 'euclidean':
            distance_metric = cv.NORM_L2
            descriptors_dtype = np.float32
        elif distance_type == 'hamming':
            distance_metric = cv.NORM_HAMMING
            descriptors_dtype = np.uint8
        else:
            raise NotImplementedError(
                'The specified distance type is not implemented')

        opencv_obj = cv.BFMatcher(
            normType=distance_metric, crossCheck=self.is_mutual)

        matches = opencv_obj.knnMatch(
            descriptors_im1.astype(descriptors_dtype),
            descriptors_im2.astype(descriptors_dtype),
            k=2)

        ratio_test_results = [
            m[0].distance <= m[1].distance*self.ratio_test_threshold
            if m is not None and len(m) > 1 else False for m in matches
        ]

        matches_after_ratio_test = [
            matches[idx][0] for idx, res in enumerate(ratio_test_results) if res]

        matches_after_ratio_test.sort(key=lambda r: r.distance)

        return np.array(
            [[m.queryIdx, m.trainIdx] for m in matches_after_ratio_test]
        ).astype(np.uint32)

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

        #  # handle NaNs
        # valid_idx_im1 = np.nonzero(~(np.isnan(descriptors_im1).any(axis=1)))[0]
        # valid_idx_im2 = np.nonzero(~(np.isnan(descriptors_im2).any(axis=1)))[0]

        # if we are not enforcing bijection, we want the descriptors with
        # larger number of entries to be used for each entry picked atmost-once.

        match_indices = None
        if not self.is_bijection:
            if descriptors_im1.shape[0] >= descriptors_im2.shape[0]:
                match_indices = self.__matcher_function(
                    descriptors_im1, descriptors_im2, distance_type)

            else:
                match_indices = self.__matcher_function(
                    descriptors_im2, descriptors_im1, distance_type)

                if match_indices.size:
                    match_indices = match_indices[:, [1, 0]]

        else:
            match_indices = self.__matcher_function(
                descriptors_im1, descriptors_im2)

            match_indices = self.__enforce_bijection(match_indices)

        return match_indices
