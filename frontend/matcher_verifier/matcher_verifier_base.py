"""Base class for the M+V (matching+verification) stage of the frontend.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed


class MatcherVerifierBase(metaclass=abc.ABCMeta):
    """Base class for all methods which provide a joint matching plus
    verification API.

    The API taking features and their descriptors from two images as input and
    returns the computed geometry (F/E matrix) and the verified correspondences.
    """

    @abc.abstractmethod
    def match_and_verify(self,
                         features_im1: np.ndarray,
                         features_im2: np.ndarray,
                         descriptors_im1: np.ndarray,
                         descriptors_im2: np.ndarray,
                         image_shape_im1: Tuple[int, int],
                         image_shape_im2: Tuple[int, int],
                         camera_instrinsics_im1: np.ndarray = None,
                         camera_instrinsics_im2: np.ndarray = None,
                         distance_type: str = 'euclidean'
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Matches the features (using their corresponding descriptors) to
        return geometrically verified outlier-free correspondences as indices of
        input features.

        Note:
        1. The function computes the fundamental matrix if intrinsics are not
           provided. Otherwise, it computes the essential matrix.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corr. descriptors from image #1
            descriptors_im2 (np.ndarray): corr. descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: index of the input features which are verified (Nx2)
        """

    def match_and_verify_and_get_features(
        self,
        features_im1: np.ndarray,
        features_im2: np.ndarray,
        descriptors_im1: np.ndarray,
        descriptors_im2: np.ndarray,
        image_shape_im1: Tuple[int, int],
        image_shape_im2: Tuple[int, int],
        camera_instrinsics_im1: np.ndarray = None,
        camera_instrinsics_im2: np.ndarray = None,
        distance_type: str = 'euclidean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calls the match_and_verify function to return actual features
        instead of indices.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corr. descriptors from image #1
            descriptors_im2 (np.ndarray): corr. descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2
            camera_instrinsics_im1 (np.ndarray, optional): Camera intrinsics
                matrix for image #1. Defaults to None.
            camera_instrinsics_im2 (np.ndarray, optional): Camera intrinsics
                matris for image #2. Default to None
            distance_type (str, optional): the space to compute the distance
                                           between descriptors. Defaults to
                                           'euclidean'.

        Returns:
            np.ndarray: estimated fundamental/essential matrix
            np.ndarray: verified features from image #1
            np.ndarray: corresponding verified features from image #2
        """
        geometry, verified_indices = self.match_and_verify(
            features_im1, features_im2,
            descriptors_im1, descriptors_im2,
            image_shape_im1, image_shape_im2,
            camera_instrinsics_im1, camera_instrinsics_im2,
            distance_type)

        return geometry, features_im1[verified_indices[:, 0], :2], \
            features_im2[verified_indices[:, 1], :2]

    def create_computation_node(self,
                                detection_description_node_im1: Delayed,
                                detection_description_node_im2: Delayed,
                                shape_node_im1: Delayed,
                                shape_node_im2: Delayed,
                                intrinsics_node_im1: Delayed,
                                intrinsics_node_im2: Delayed
                                ) -> Delayed:
        """Create a computation node for processing a pair of image.

        Args:
            detection_description_node_im1 (Delayed): [description]
            detection_description_node_im2 (Delayed): [description]
            shape_node_im1 (Delayed): [description]
            shape_node_im2 (Delayed): [description]
            intrinsics_node_im1 (Delayed): [description]
            intrinsics_node_im2 (Delayed): [description]

        Returns:
            Delayed: [description]
        """
        return dask.delayed(self.match_and_verify_and_get_features)(
            detection_description_node_im1[0],
            detection_description_node_im2[0],
            detection_description_node_im1[1],
            detection_description_node_im2[1],
            shape_node_im1,
            shape_node_im2,
            intrinsics_node_im1,
            intrinsics_node_im2
        )

    def create_computation_graph(self,
                                 pair_indices: List[Tuple[int, int]],
                                 detection_description_graph: List[Delayed],
                                 image_shape_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 distance_type: str = 'euclidean'
                                 ) -> Dict[Tuple[int, int], Delayed]:
        """Create computation graph for the performing matching and verification
        on results from detector-descriptor.

        Args:
            detection_description_graph (List[Delayed]): computation graph from
                                                        detector descriptor.
            loader_graph (List[Delayed]): computation graph from loader.

        Returns:
            Dict[Tuple[int, int], Delayed]: delayed dask tasks for pairs of
                                            inputs.
        """
        graph = dict()

        for idx1, idx2 in pair_indices:

            graph_component_im1 = detection_description_graph[idx1]
            graph_component_im2 = detection_description_graph[idx2]

            graph[(idx1, idx2)] = \
                dask.delayed(self.match_and_verify_and_get_features)(
                    graph_component_im1[0], graph_component_im2[0],
                    graph_component_im1[1], graph_component_im2[1],
                    image_shape_graph[idx1], image_shape_graph[idx2],
                    camera_intrinsics_graph[idx1],
                    camera_intrinsics_graph[idx2],
                    distance_type
            )

        return graph
