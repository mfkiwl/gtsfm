"""RootSIFT descriptor implementation.

This descriptor was proposed in 'Three things everyone should know to improve
object retrieval' and is build upon OpenCV's SIFT descriptor.

Note: this is a descriptor

References:
- https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
- https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

Authors: Ayush Baid
"""
import numpy as np

from common.image import Image
from frontend.descriptor.sift import SIFTDescriptor


class RootSIFTDescriptor(SIFTDescriptor):
    """RootSIFT descriptor using OpenCV's implementation."""

    def describe(self, image: Image, features: np.ndarray) -> np.ndarray:
        """Assign descriptors to detected features in an image.

        Arguments:
            image (Image): the input image
            features (np.ndarray): the features to describe

        Returns:
            np.ndarray: the descriptors for the input features
        """

        if features.size == 0:
            return np.array([])

        descriptors = super().describe(image, features)

        # Step 1: L1 normalization
        descriptors = descriptors / \
            (np.sum(descriptors, axis=1, keepdims=True)+1e-8)

        # Step 2: Element wise square-root
        descriptors = np.sqrt(descriptors)

        # Step 3: L2 normalization
        descriptors = descriptors / \
            (np.linalg.norm(descriptors, axis=1, keepdims=True)+1e-8)

        return descriptors
