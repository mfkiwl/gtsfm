""" Base class for Loaders.

Authors: Frank Dellaert and Ayush Baid
"""

import abc
from typing import List, Tuple

import dask
import numpy as np
from dask.delayed import Delayed

from common.image import Image


class LoaderBase(metaclass=abc.ABCMeta):
    """Base class for Loaders.

    The loader provides APIs to get an image, either directly or as a dask delayed task
    """

    # ignored-abstractmethod
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset

        Returns:
            int: the number of images
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_image(self, index: int) -> Image:
        """
        Get the image at the given index

        Args:
            index (int): the index to fetch

        Returns:
            Image: the image at the query index
        """

    # ignored-abstractmethod
    @abc.abstractmethod
    def get_camera_intrinsics(self, index: int) -> np.ndarray:
        """Get the camera intrinsics at the given index.

        Args:
            index (int): the index to fetch

        Returns:
            np.ndarray: the 3x3 intrinsics matrix of the camera
        """

    @abc.abstractmethod
    def get_geometry(self, idx1: int, idx2: int) -> np.ndarray:
        """Get the ground truth fundamental matrix/homography from idx1 to idx2.

        The function returns either idx1_F_idx2 or idx1_H_idx2

        Args:
            idx1 (int): one of the index
            idx2 (int): one of the index

        Returns:
            np.ndarray: fundamental matrix/homograph matrix
        """

    @abc.abstractmethod
    def validate_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Args:
            idx1 (int): first index of the pair.
            idx2 (int): second index of the pair.

        Returns:
            bool: validation result.
        """

    def delayed_get_image(self, index: int) -> Delayed:
        """
        Wraps the get_image evaluation in a dask.delayed

        Args:
            index (int): the image index

        Returns:
            Delayed: the get_image function for the given index wrapped in dask.delayed
        """
        return dask.delayed(self.get_image)(index)

    def image_load_graph(self) -> List[Delayed]:
        """Creates the computation graph for all image fetches.

        Returns:
            List[Delayed]: delayed tasks for loading the image
        """

        return [dask.delayed(self.get_image)(x) for x in range(self.__len__())]

    def image_shape_graph(self) -> List[Delayed]:
        """Creates the computation graph for all image shapes

        Returns:
            List[Delayed]: delayed tasks for loading the image
        """

        return [(dask.delayed(self.get_image)(x)).shape
                for x in range(self.__len__())]

    def intrinsics_graph(self) -> List[Delayed]:
        """Creates the computation graph for camera intrinsics.

        Returns:
            List[Delayed]: delayed tasks for intrinsics.
        """
        N = self.__len__()

        return [dask.delayed(self.get_camera_intrinsics)(x) for x in range(N)]

    def get_valid_pairs(self) -> List[Tuple[int, int]]:
        """Get the valid pairs of images for this loader.

        Returns:
            List[Tuple[int, int]]: valid index pairs
        """
        indices = []

        for idx1 in range(self.__len__()):
            for idx2 in range(self.__len__()):
                if(self.validate_pair(idx1, idx2)):
                    indices.append((idx1, idx2))

        return indices
