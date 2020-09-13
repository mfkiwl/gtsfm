"""The front-end (detection-description-matching-verification) implementation.

This class combines the different components of front-ends and provides a
function to generate geometrically verified correspondences and pose-information
between pairs of images.

Authors: Ayush Baid
"""
from typing import Dict, Tuple

import numpy as np
from dask.delayed import Delayed

from common.image import Image
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase
from frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from loader.loader_base import LoaderBase


class FrontEndWrapper:
    """The complete front-end class (composed on different modules)."""

    def __init__(self,
                 detector_descriptor: DetectorDescriptorBase,
                 matcher_verifier: MatcherVerifierBase):
        """Initializes the front-end using different modules.

        Args:
            detector_descriptor (DetectorDescriptorBase): Detection-description
                                                          object
            matcher_verifier (MatcherVerifierBase): Matching-Verification object
        """
        self.detector_descriptor = detector_descriptor
        self.matcher_verifier = matcher_verifier

    def descriptor_distance_type(self, descriptors: np.ndarray) -> str:
        if descriptors is None:
            # hack
            return 'euclidean'

        return 'hamming' if descriptors.dtype == np.bool else 'euclidean'

    def run_single_pair(self,
                        image_1: Image,
                        image_2: Image,
                        camera_intrinsics_1: np.ndarray = None,
                        camera_intrinsics_2: np.ndarray = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the front-end on a pair of images.

        Note:
        1. When intrinsics are supplied, the essential matrix is computed.
          Otherwise, the fundamental matrix is computed.

        Args:
            image_1 (Image): image #1
            image_2 (Image): image #2
            camera_intrinsics_1 (np.ndarray, optional): Intrinsics for camera
                                                        for image_1. Defaults to
                                                        None.
            camera_intrinsics_2 (np.ndarray, optional): Intrinsics for camera
                                                        for image_2. Defaults to
                                                        None.

        Returns:
            np.ndarray: fundamental/essential matrix
            np.ndarray: verified correspondences from image #1
            np.ndarray: verified correspondences from image #2
        """

        features_1, descriptors_1 = \
            self.detector_descriptor.detect_and_describe(image_1)

        features_2, descriptors_2 = \
            self.detector_descriptor.detect_and_describe(image_2)

        return self.matcher_verifier.match_and_verify_and_get_features(
            features_1,
            features_2,
            descriptors_1,
            descriptors_2,
            image_1.shape,
            image_2.shape,
            camera_intrinsics_1,
            camera_intrinsics_2,
            self.descriptor_distance_type(descriptors_1)
        )

    def run_loader(self,
                   loader: LoaderBase
                   ) -> Dict[
                       Tuple[int, int],
                       Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run the front-end on a loader.

        Note:
        1. When intrinsics are supplied by the loader, the essential matrix is
           computed. Otherwise, the fundamental matrix is computed.

        Args:
            loader (LoaderBase): loader for the dataset

        Returns:
            Dict[ Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray] ]: 
                front-end results for all valid image pairs
        """

        features_list = []
        descriptors_list = []

        for idx in range(len(loader)):
            features, descriptors = \
                self.detector_descriptor.detect_and_describe(
                    loader.get_image(idx)
                )

            features_list.append(features)
            descriptors_list.append(descriptors)

        results = dict()

        for idx1, idx2 in loader.get_valid_pairs():
            results[(idx1, idx2)] = \
                self.matcher_verifier.match_and_verify_and_get_features(
                    features_list[idx1],
                    features_list[idx2],
                    descriptors_list[idx1],
                    descriptors_list[idx2],
                    loader.get_image(idx1).shape,
                    loader.get_image(idx2).shape,
                    loader.get_camera_intrinsics(idx1),
                    loader.get_camera_intrinsics(idx2)
            )

        return results

    def create_computation_graph(self,
                                 loader: LoaderBase
                                 ) -> Dict[Tuple[int, int], Delayed]:
        """Create the frontend computation graph for a loader graph.

        Args:
            loader (LoaderBase): loader for the dataset

        Returns:
            Dict[Tuple[int, int], Delayed]: delayed task tasks for pairs of
                                            input.
        """

        detection_description_graph = \
            self.detector_descriptor.create_computation_graph(
                loader.image_load_graph()
            )

        frontend_graph = self.matcher_verifier.create_computation_graph(
            loader.get_valid_pairs(),
            detection_description_graph,
            loader.image_shape_graph(),
            loader.intrinsics_graph()
        )

        return frontend_graph
