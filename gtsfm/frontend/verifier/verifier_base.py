"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints


NUM_MATCHES_REQ_E_MATRIX = 5
NUM_MATCHES_REQ_F_MATRIX = 8


class VerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the estimated essential matrix as well as
    geometrically verified points.
    """

    def __init__(self, use_intrinsics_in_verification: bool):
        """Initializes the verifier.

        Args:
            use_intrinsics_in_verification: Flag to perform keypoint normalization and compute the essential matrix 
                                            instead of fundamental matrix. This should be preferred when the exact
                                            intrinsics are known as opposed to approximating them from exif data.
        """
        self._use_intrinsics_in_verification = use_intrinsics_in_verification
        self._min_matches = (
            NUM_MATCHES_REQ_E_MATRIX if self._use_intrinsics_in_verification else NUM_MATCHES_REQ_F_MATRIX
        )

        self._failure_result = (None, None, np.array([], dtype=np.uint64))

    @abc.abstractmethod
    def verify(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Performs verification of correspondences between two images to recover the relative pose and indices of
        verified correspondences.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3. These are subset of match_indices.
        """

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        matches_i1i2_graph: Delayed,
        intrinsics_i1_graph: Delayed,
        intrinsics_i2_graph: Delayed,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Generates the computation graph to perform verification of putative correspondences.

        Args:
            image_pair_indices: 2-tuple (i1,i2) specifying image pair indices
            detection_graph: nodes with features for each image.
            matcher_graph: nodes with matching results for pairs of images.
            camera_intrinsics_graph: nodes with intrinsics for each image.

        Returns:
            Delayed dask task for rotation i2Ri1 for specific image pair.
            Delayed dask task for unit translation i2Ui1 for specific image pair.
            Delayed dask task for indices of verified correspondence indices for the specific image pair
        """
        # we cannot immediately unpack the result tuple, per dask syntax
        result = dask.delayed(self.verify)(
            keypoints_i1_graph, keypoints_i2_graph, matches_i1i2_graph, intrinsics_i1_graph, intrinsics_i2_graph,
        )
        i2Ri1_graph = result[0]
        i2Ui1_graph = result[1]
        v_corr_idxs_graph = result[2]

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph
