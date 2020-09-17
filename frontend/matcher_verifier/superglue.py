"""SuperGlue matcher+verifier implementation

The network was proposed in 'SuperGlue: Learning Feature Matching with Graph
Neural Networks' and is implemented by wrapping over author's source-code.

Note: the pretrained model only supports superpoint right now.

References:
- http://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
- https://github.com/magicleap/SuperGluePretrainedNetwork

Authors: Ayush Baid
"""
from typing import Tuple

import numpy as np
import torch

from frontend.matcher_verifier.matcher_verifier_base import MatcherVerifierBase
from frontend.verifier.simple_verifier import SimpleVerifier
from frontend.verifier.verifier_base import VerifierBase
from implementation.superglue.models.superglue import SuperGlue


class SuperGlueMatcherVerifier(MatcherVerifierBase):
    """SuperGlue matcher+verifier implementation."""
    # TODO: handle variable descriptor dimensions

    def __init__(self, is_cuda=True,
                 post_process_verifier: VerifierBase = SimpleVerifier()):
        """Initialise the configuration and the parameters."""

        self.post_process_verifier = post_process_verifier

        config = {
            'descriptor_dim': 256,
            'weights_path': '../models/superglue/superglue_outdoor.pth'
        }

        self.use_cuda = is_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = SuperGlue(config).to(self.device)

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

        F, indices, _, _ = self.__compute(features_im1,
                                          features_im2,
                                          descriptors_im1,
                                          descriptors_im2,
                                          image_shape_im1,
                                          image_shape_im2)[:2]

        return F, indices

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
            distance_type: str = 'euclidean'):
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

        F, _, verified_features_im1, verified_features_im2 = self.__compute(
            features_im1,
            features_im2,
            descriptors_im1,
            descriptors_im2,
            image_shape_im1,
            image_shape_im2
        )

        return F, verified_features_im1, verified_features_im2

    def __compute(self,
                  features_im1: np.ndarray,
                  features_im2: np.ndarray,
                  descriptors_im1: np.ndarray,
                  descriptors_im2: np.ndarray,
                  image_shape_im1: Tuple[int, int],
                  image_shape_im2: Tuple[int, int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Common function which performs computations for both the public APIs.

        Args:
            features_im1 (np.ndarray): features from image #1
            features_im2 (np.ndarray): features from image #2
            descriptors_im1 (np.ndarray): corresponding descriptors from image #1
            descriptors_im2 (np.ndarray): corresponding descriptors from image #2
            image_shape_im1 (Tuple[int, int]): size of image #1
            image_shape_im2 (Tuple[int, int]): size of image #2

        Returns:
            np.ndarray: estimated fundamental matrix
            np.ndarray: verified correspondences as index of the input features in a Nx2 array
            np.ndarray: features from image #1 corresponding to verified match indices
            np.ndarray: features from image #2 corresponding to verified match indices

        """

        if features_im1.size == 0 or features_im2.size == 0:
            return None, np.array([], dtype=np.uint32), np.array([]), np.array([])

        if descriptors_im1.size == 0 or \
                descriptors_im2.size == 0 or \
                descriptors_im1 is None or \
                descriptors_im2 is None:
            return None, np.array([], dtype=np.uint32), np.array([]), np.array([])

        if features_im1.shape[1] < 4 or features_im2.shape[1] < 4:
            # we do not have the feature confidence as input
            raise Exception("No confidence score on detected features")

        if descriptors_im1.shape[1] != 256 or descriptors_im2.shape[1] != 256:
            print(descriptors_im1.shape)
            raise Exception(
                "Superglue pretrained network only works on 256 dimensional descriptors"
            )

        # convert to datatypes required by the forward function
        data = {
            'keypoints0': torch.from_numpy(np.expand_dims(features_im1[:, :2], 0)).to(self.device),
            'keypoints1': torch.from_numpy(np.expand_dims(features_im2[:, :2], 0)).to(self.device),
            'descriptors0': torch.from_numpy(np.expand_dims(np.transpose(descriptors_im1), 0)).to(self.device),
            'descriptors1': torch.from_numpy(np.expand_dims(np.transpose(descriptors_im2), 0)).to(self.device),
            'scores0': torch.from_numpy(np.expand_dims(features_im1[:, 3], (0))).to(self.device),
            'scores1': torch.from_numpy(np.expand_dims(features_im2[:, 3], (0))).to(self.device),
            'image_shape1': image_shape_im1,
            'image_shape2': image_shape_im2
        }

        superglue_results = self.model(data)

        matches_for_features_im1 = np.squeeze(
            superglue_results['matches0'].detach().cpu().numpy())

        match_indices_im1 = np.where(matches_for_features_im1 > -1)[0]
        match_indices_im2 = matches_for_features_im1[match_indices_im1]

        verified_indices = np.concatenate(
            [match_indices_im1.reshape(-1, 1), match_indices_im2.reshape(-1, 1)], axis=1).astype(np.uint32)

        verified_features_im1 = features_im1[verified_indices[:, 0], :2]
        verified_features_im2 = features_im2[verified_indices[:, 1], :2]

        geometry_matrix, inner_inlier_idx = self.post_process_verifier.verify(
            verified_features_im1,
            verified_features_im2,
            image_shape_im1,
            image_shape_im2,
            None,
            None)

        return geometry_matrix, \
            verified_indices[inner_inlier_idx], \
            verified_features_im1[inner_inlier_idx], \
            verified_features_im2[inner_inlier_idx]
