"""Simple non-robust verifier using OpenCV.

This verifier uses the 5-point algorithm proposed in "An efficient solution to
the five-point relative pose problem" for essential matrix computation and
8-point algorithm's version proposed in "In defence of the 8-point algorithm"
for fundamental matrix computation.
References:
- https://ieeexplore.ieee.org/document/1288525
- https://ieeexplore.ieee.org/document/601246
- https://docs.opencv.org/4.5.0/d9/d0c/group__calib3d.html


Authors: Ayush Baid
"""
import logging
import sys
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.verification as verification_utils
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.common.keypoints import Keypoints

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# minimum matches required for computing the F-matrix
NUM_MATCHES_REQ_F_MATRIX = 8
NUM_MATCHES_REQ_E_MATRIX = 5


class SimpleVerifier(VerifierBase):
    def __init__(self):
        super().__init__(min_pts=NUM_MATCHES_REQ_E_MATRIX)

    def verify_with_exact_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are known. The
        feature coordinates are normalized and the essential matrix is directly
        estimated.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        logging.warning(
            "5-point algorithm has not been implemented yet."
            + "Falling back to 8-point algorithm for fundamental matrix"
        )

        return self.verify_with_approximate_intrinsics(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

    def verify_with_approximate_intrinsics(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Estimates the essential matrix and verifies the feature matches.

        Note: this function is preferred when camera intrinsics are approximate
        (i.e from image size/exif). The feature coordinates are used to compute
        the fundamental matrix, which is then converted to the essential matrix.

        Args:
            keypoints_i1: detected features in image #i1.
            keypoints_i2: detected features in image #i2.
            match_indices: matches as indices of features from both images, of
                           shape (N3, 2), where N3 <= min(N1, N2).
            camera_intrinsics_i1: intrinsics for image #i1.
            camera_intrinsics_i2: intrinsics for image #i2.

        Returns:
            Estimated rotation i2Ri1, or None if it cannot be estimated.
            Estimated unit translation i2Ui1, or None if it cannot be estimated.
            Indices of verified correspondences, of shape (N, 2) with N <= N3.
                These indices are subset of match_indices.
        """
        verified_indices = np.array([], dtype=np.uint32)

        # check if we don't have the minimum number of points
        if match_indices.shape[0] < NUM_MATCHES_REQ_F_MATRIX:
            return None, None, verified_indices

        i2Fi1, mask = cv.findFundamentalMat(
            keypoints_i1.coordinates[match_indices[:, 0]],
            keypoints_i2.coordinates[match_indices[:, 1]],
            cv.FM_8POINT,
        )

        inlier_idxes = np.where(mask.ravel() == 1)[0]

        i2Ei1_matrix = verification_utils.fundamental_to_essential_matrix(
            i2Fi1, camera_intrinsics_i1, camera_intrinsics_i2
        )

        (
            i2Ri1,
            i2Ui1,
        ) = verification_utils.recover_relative_pose_from_essential_matrix(
            i2Ei1_matrix,
            keypoints_i1.coordinates[match_indices[inlier_idxes, 0]],
            keypoints_i2.coordinates[match_indices[inlier_idxes, 1]],
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        return i2Ri1, i2Ui1, match_indices[inlier_idxes]
