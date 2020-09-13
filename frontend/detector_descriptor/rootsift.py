"""RootSIFT detector-descriptor implementation.

This descriptor was proposed in 'Three things everyone should know to improve
object retrieval' and is build upon OpenCV's SIFT descriptor.

References:
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np

from common.image import Image
from frontend.detector_descriptor.sift import SIFTDetectorDescriptor


class RootSIFTDetectorDescriptor(SIFTDetectorDescriptor):
    """RootSIFT detector-descriptor using OpenCV's implementation."""

    def detect_and_describe(self,
                            image: Image) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature detection as well as their description in a single
        step.

        Refer to detect() in DetectorBase and describe() in DescriptorBase for
        details about the output format.

        Args:
            image (Image): the input image

        Returns:
            Tuple[np.ndarray, np.ndarray]: detected features and their
                                           descriptions as two numpy arrays
        """
        features, descriptors = super().detect_and_describe(image)

        # Step 1: L1 normalization
        descriptors = descriptors / \
            (np.sum(descriptors, axis=1, keepdims=True)+1e-8)

        # Step 2: Element wise square-root
        descriptors = np.sqrt(descriptors)

        # Step 3: L2 normalization
        descriptors = descriptors / \
            (np.linalg.norm(descriptors, axis=1, keepdims=True)+1e-8)

        return features, descriptors
