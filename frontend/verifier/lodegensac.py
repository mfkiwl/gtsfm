"""Locally Optimized (LO) RANSAC + Degensac verifier.

LO-Ransac was proposed in 'Locally Optimized RANSAC' and Degensac was proposed
'Two-view Geometry Estimation Unaffected by a Dominant Plane'. This
implementation is done by wrapping over a third-party module.

Authors: Ayush Baid
"""

from typing import Tuple

import numpy as np
import pydegensac

from frontend.verifier.verifier_base import VerifierBase


class LODegensac(VerifierBase):
    """Locally Optimized (LO) RANSAC + Degensac verifier."""

    def __init__(self, dist_threshold: float = 0.5):
        super().__init__(min_pts=8)

        self.dist_threshold = dist_threshold

    def verify(self,
               matched_features_im1: np.ndarray,
               matched_features_im2: np.ndarray,
               image_shape_im1: Tuple[int, int],
               image_shape_im2: Tuple[int, int],
               camera_instrinsics_im1: np.ndarray = None,
               camera_instrinsics_im2: np.ndarray = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform the geometric verification of the matched features.

        Note:
        1. The number of input features from image #1 and image #2 are equal.
        2. The function computes the fundamental matrix if intrinsics are not
           provided. Otherwise, it computes the essential matrix.

        Args:
            matched_features_im1 (np.ndarray): matched features from image #1
            matched_features_im2 (np.ndarray): matched features from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the input matches which are verified
        """
        if camera_instrinsics_im1 is not None or \
                camera_instrinsics_im2 is not None:
            raise NotImplementedError(
                "Degensac for essential matrix is not implemented")

        if matched_features_im1.shape[0] < self.min_pts:
            return None, np.array([]).astype(np.uint32)

        F, mask = pydegensac.findFundamentalMatrix(
            matched_features_im1[:, :2],
            matched_features_im2[:, :2],
            px_th=self.dist_threshold
        )

        inlier_idx = np.where(mask.ravel())[0]

        return F, inlier_idx
