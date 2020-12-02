"""Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import cv2 as cv
import numpy as np

from common.keypoints import Keypoints


def keypoints_from_array(features: np.ndarray) -> List[cv.KeyPoint]:
    """Converts the features from numpy array to cv keypoints.

    Args:
        features: Numpy array of shape (N,2+) representing feature points.

    Returns:
        OpenCV KeyPoint objects representing the given features.
    """
    # TODO(ayush): what should be scale if not provided?

    # input features is a 2D array
    if features.shape[1] < 3:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=2) for f in features]
    elif features.shape[1] < 4:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=float([2])) for f in features]
    else:
        keypoints = [cv.KeyPoint(
            x=float(f[0]),
            y=float(f[1]),
            _size=float(f[2]),
            _response=float(f[3])) for f in features]

    return keypoints


def cast_to_gtsfm_keypoints(keypoints: List[cv.KeyPoint]) -> Keypoints:
    """Cast list of OpenCV's keypoints to GTSFM's keypoints.

    Args:
        keypoints: list of OpenCV's keypoints.

    Returns:
        GTSFM's keypoints with the same information as input keypoints.
    """
    coordinates = []
    scales = []
    responses = []
    for kp in keypoints:
        coordinates.append([kp.pt[0], kp.pt[1]])
        scales.append(kp.size)
        responses.append(kp.response)

    return Keypoints(coordinates=np.array(coordinates),
                     scales=np.array(scales),
                     responses=np.array(responses))
