"""Base class for the V (verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed


class VerifierBase(metaclass=abc.ABCMeta):
    """Base class for all verifiers.

    Verifiers take the coordinates of the matches as inputs and returns the
    estimated fundamental matrix as well as geometrically verified points.
    """

    def __init__(self, min_pts):
        self.min_pts = min_pts

    @abc.abstractmethod
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
            image_shape_im1 (Tuple[int, int]): shape of image #1
            image_shape_im2 (Tuple[int, int]): shape of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the input matches which are verified
        """

    def verify_and_get_features(
            self,
            matched_features_im1: np.ndarray,
            matched_features_im2: np.ndarray,
            image_shape_im1: Tuple[int, int],
            image_shape_im2: Tuple[int, int],
            camera_instrinsics_im1: np.ndarray = None,
            camera_instrinsics_im2: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs geometrics verification using verify function and returns
        actual features instead of indices.

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
            np.ndarray: verified features from image #1
            np.ndarray: corresponding verified features from image #2
        """
        geometry, verified_indices = self.verify(
            matched_features_im1,
            matched_features_im2,
            image_shape_im1,
            image_shape_im2,
            camera_instrinsics_im1,
            camera_instrinsics_im2
        )

        return geometry, \
            matched_features_im1[verified_indices], \
            matched_features_im2[verified_indices]

    def create_computation_graph(self,
                                 matcher_graph: Dict[Tuple[int, int], Delayed],
                                 image_shape_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 ) -> Dict[Tuple[int, int], Delayed]:
        """Creates computation graph for performing verification on results
        from matcher.

        Args:
            matcher_graph (Dict[Tuple[int, int], Delayed]): computation graph
                                                            from matcher
            loader_graph (List[Delayed]): computation graph from loader

        Returns:
            Dict[Tuple[int, int], Delayed]: delayed dask tasks for verificatrion
        """

        result = dict()

        for image_idx_tuple, matcher_task in matcher_graph.items():

            image_shape_tasks = image_shape_graph[image_idx_tuple]
            camera_intrinsics_tasks = camera_intrinsics_graph[image_idx_tuple]

            result[image_idx_tuple] = dask.delayed(self.verify_and_get_features)(
                matcher_task[0],
                matcher_task[1],
                image_shape_tasks[0],
                image_shape_tasks[1],
                camera_intrinsics_tasks[0],
                camera_intrinsics_tasks[1],
            )

        return result
