"""RANSAC based verification implementation.

This method was proposed in '<>' and is implemented by wrapping over OpenCV's
API.

Authors: Ayush Baid
"""
from typing import Tuple

import cv2 as cv
import numpy as np

import utils.verification as verification_utils
from frontend.verifier.verifier_base import VerifierBase


class RANSAC(VerifierBase):
    """Ransac Verifier."""

    def __init__(self, probability: float = 0.99, dist_threshold: float = 0.5):
        super().__init__(min_pts=8)

        self.probability = probability
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
        geometry_matrix = None
        verified_indices = np.array([], dtype=np.uint32)

        if matched_features_im1.shape[0] < self.min_pts:
            return geometry_matrix, verified_indices

        if camera_instrinsics_im1 is not None and \
                camera_instrinsics_im2 is not None:
            # compute the essential matrix'
            normalized_features_im1 = verification_utils.normalize_coordinates(
                matched_features_im1[:, :2], camera_instrinsics_im1)

            normalized_features_im2 = verification_utils.normalize_coordinates(
                matched_features_im2[:, :2], camera_instrinsics_im2)

            geometry_matrix, mask = cv.findEssentialMat(
                normalized_features_im1[:, :2],
                normalized_features_im2[:, :2],
                np.eye(3),
                cv.RANSAC,
                prob=self.probability,
                threshold=self.dist_threshold
            )

            verified_indices = np.where(mask.ravel() == 1)[0]
        else:
            geometry_matrix, mask = cv.findFundamentalMat(
                matched_features_im1[:, :2],
                matched_features_im2[:, :2],
                cv.FM_RANSAC,
                confidence=self.probability,
                ransacReprojThreshold=self.dist_threshold
            )

            verified_indices = np.where(mask.ravel() == 1)[0]

        return geometry_matrix, verified_indices
