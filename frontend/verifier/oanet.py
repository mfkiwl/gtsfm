"""Order-Aware Net verifier implementation.

The verifier was proposed in 'Learning Two-View Correspondences and Geometry
Using Order-Aware Network' and is implemented by wrapping over the authors'
source-code.

References:
- https://arxiv.org/abs/1908.04964 
- https://github.com/zjhthu/OANet

Authors: Ayush Baid
"""

import os
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch

import utils.verification as verification_utils
from frontend.verifier.verifier_base import VerifierBase
from thirdparty.implementation.oanet.core.oan import OANet


class OANetVerifier(VerifierBase):
    """OA-Net Verifier."""

    def __init__(self,
                 is_cuda=True,
                 post_process_verifier: VerifierBase = None):
        super().__init__(min_pts=8)

        is_cuda = is_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda else "cpu")

        self.post_process_verifier = post_process_verifier

        model_path = os.path.abspath(os.path.join(
            'thirdparty', 'models', 'oanet', 'gl3d', 'sift-4000', 'model_best.pth'))

        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = 12
        self.default_config['clusters'] = 500
        self.default_config['use_ratio'] = 0  # not using ratio
        self.default_config['use_mutual'] = 0  # not using mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = 1
        default_config_ = namedtuple('Config', self.default_config.keys())(
            *self.default_config.values())

        self.model = OANet(default_config_, self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

        if is_cuda:
            self.model = self.model.cuda()

        self.model.eval()

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

        return_fundamental = camera_instrinsics_im1 is None and \
            camera_instrinsics_im2 is None

        if camera_instrinsics_im1 is None:
            camera_instrinsics_im1 = \
                verification_utils.intrinsics_from_image_shape(image_shape_im1)

        if camera_instrinsics_im2 is None:
            camera_instrinsics_im2 = \
                verification_utils.intrinsics_from_image_shape(image_shape_im2)

        # normalize the features
        normalized_features_im1 = verification_utils.normalize_coordinates(
            matched_features_im1[:, :2], camera_instrinsics_im1)[:, :2]

        normalized_features_im2 = verification_utils.normalize_coordinates(
            matched_features_im2[:, :2], camera_instrinsics_im2)[:, :2]
        # normalized_features_im1 = matched_features_im1[:, :2]
        # normalized_features_im2 = matched_features_im2[:, :2]

        if matched_features_im1.shape[0] < self.min_pts:
            return geometry_matrix, verified_indices

        with torch.no_grad():
            normalized_keypoints = [
                torch.from_numpy(self.__normalize_kpts(
                    normalized_features_im1).astype(np.float32)).to(self.device),
                torch.from_numpy(self.__normalize_kpts(
                    normalized_features_im2).astype(np.float32)).to(self.device)
            ]

            corr = torch.cat(normalized_keypoints, dim=-1)

            corr = corr.unsqueeze(0).unsqueeze(0)

            data = {}
            data['xs'] = corr
            data['sides'] = []

            try:
                y_hat, e_hat = self.model(data)
            except RuntimeError as e:
                if str(e) != 'symeig_cpu: the algorithm failed to converge; 8 off-diagonal elements of an intermediate tridiagonal form did not converge to zero.':
                    raise
                else:
                    return geometry_matrix, verified_indices

            y = y_hat[-1][0, :].cpu().numpy()
            geometry_matrix = e_hat[-1][0, :].cpu().numpy().reshape(3, 3)
            verified_indices = np.where(
                y > self.default_config['inlier_threshold'])[0].astype(
                    np.uint32)

            if return_fundamental:
                geometry_matrix = camera_instrinsics_im2.T @ \
                    geometry_matrix @ camera_instrinsics_im1

        if self.post_process_verifier is not None:
            geometry_matrix, new_indices = self.post_process_verifier.verify(
                matched_features_im1[verified_indices],
                matched_features_im2[verified_indices],
                image_shape_im1,
                image_shape_im2,
                None if return_fundamental else camera_instrinsics_im1,
                None if return_fundamental else camera_instrinsics_im2
            )

            verified_indices = verified_indices[new_indices]

        return geometry_matrix, verified_indices

    def __normalize_kpts(self, kpts):
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3, 3])
        T[0, 0], T[1, 1], T[2, 2] = scale, scale, 1
        T[0, 2], T[1, 2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + \
            np.array([T[0, 2], T[1, 2]])
        return nkpts
