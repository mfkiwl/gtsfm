""" 
Common utilities for feature points.

Authors: Ayush Baid
"""
from typing import List

import numpy as np
import cv2 as cv

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


def convert_to_homogenous(normal_coordinates: np.array) -> np.array:
    '''
    Convert normal coordinates to homogenous coordinates

    Args:
        normal_coordinates (numpy array): Normal (x,y) 2d point coordinates with shape Nx2
    Returns
        numpy array: the corresponding homogenous coordinates. Shape=Nx3
    '''

    if np.shape(normal_coordinates)[1] == 3:
        return normal_coordinates

    return np.append(normal_coordinates, np.ones((normal_coordinates.shape[0], 1)), axis=1)


def apply_homography_transform(coordinates: np.array,
                               homography: np.array) -> np.array:
    '''
    Apply the homography transform to the given coordinates

    Args:
        coordinates (numpy array):  coordinates in image. Shape=Nx2 or Nx3
        homography (numpy array):   homography transform. Shape=3x3
    Returns:
        numpy array: transformed coordinates with the shape shape as input coordinates
    '''
    if coordinates is None or coordinates.size == 0:
        return coordinates

    return_normal = False
    if coordinates.shape[1] == 2:
        return_normal = True

    coordinates = convert_to_homogenous(coordinates)

    transformed_coordinates = np.matmul(coordinates, np.transpose(homography))

    # maintain the homogenous condition
    transformed_coordinates = np.divide(transformed_coordinates,
                                        transformed_coordinates[:, 2].reshape(-1, 1))

    if return_normal:
        transformed_coordinates = transformed_coordinates[:, :2]

    return transformed_coordinates


def overlap_mask(coordinates: np.array,
                 transformed_im_size: List[int],
                 homography: np.array):
    '''
    Generates a mask if the features are in the covisible region

    # TODO: write unit test

    Args:
        coordinates (numpy array):          the coordinates to perform the check on
        transformed_im_size (list of int):  the size of the image on which the coordinates are mapped to
        homography (numpy array):           homography transform
    Returns
        boolean numpy array: overlap mask for the coordinates
    '''
    return validate_coordinates(
        apply_homography_transform(coordinates,
                                   homography),
        transformed_im_size
    )


def get_num_points_in_overlap_homography(feature_coords_1,
                                         feature_coords_2,
                                         im_size_1,
                                         im_size_2,
                                         homography,
                                         overlap_mask_1: np.array = None,
                                         overlap_mask_2: np.array = None) -> (int, int):
    '''
    Get the number of points which fall in the covisible region

    Args:
        feature_coords_1 (numpy array): coordinates from 1st image
        feature_coords_2 (numpy array): coordinates from 2nd image
        im_size_1 (list of int):        shape of 1st image
        im_size_2 (list of int):        shape of 2nd image
        homography (numpy array):       the homography transform
        overlap_mask_1 (numpy array):   precomputed covisibility mask for feature_coords_1
        overlap_mask_2 (numpy array):   precomputed covisibility mask for feature_coords_2
    Returns:
        int:    number of points in feature_coords_1 which are in the covisible region
        int:    number of points in feature_coords_2 which are in the covisible region
    '''

    if overlap_mask_1 is None:
        overlap_mask_1 = overlap_mask(
            feature_coords_1, im_size_2, homography)

    if overlap_mask_2 is None:
        overlap_mask_2 = overlap_mask(
            feature_coords_2, im_size_1, np.linalg.inv(homography))

    return np.count_nonzero(overlap_mask_1), np.count_nonzero(overlap_mask_2)


def validate_coordinates(coordinates: np.array,
                         im_size: List[int]) -> np.array:
    '''
    Validate if the coordinates lie within the image bound

    Args:
        coordinates (numpy array):  the coordinates of the features
        im_size (list of int):      shape of the image
    Returns:
        bool numpy array: the boolean flag for each input
    '''

    # checked with Key.Net implementation but did not find their implementation

    is_valid = np.logical_and(
        coordinates >= [0, 0], coordinates <= (im_size[1], im_size[0])).all(axis=1)

    return is_valid


def convert_to_epipolar_lines(coordinates: np.array,
                              f_matrix: np.array,
                              is_normalize: bool = False) -> np.array:
    '''
    Convert the feature coordinates to epipolar lines

    Args:
        coordinates (numpy array):  coordinates of the features
        f_matrix (numpy array):     fundamental matrix
        is_normalize (bool):        flag indicating if the epipolar lines should be normalized to unit norm
    Returns:
        bool numpy array: the boolean flag for each input
    '''
    if coordinates is None or coordinates.size == 0:
        return coordinates

    # convert the feature_coordinates_1 to epipolar lines
    epipolar_lines = np.matmul(convert_to_homogenous(coordinates),
                               np.transpose(f_matrix)
                               )

    # normalize the lines
    if is_normalize:
        lines_norm = np.linalg.norm(epipolar_lines, axis=1, keepdims=True)

        epipolar_lines = np.divide(epipolar_lines, lines_norm)

    return epipolar_lines


def generate_distance_vector_homography(feature_coords_1: np.array,
                                        feature_coords_2: np.array,
                                        homography: np.array) -> np.array:
    '''
    Get the distance between the features under homography transform (row-level correspondence)

    Args:
        feature_coords_1 (numpy array): coordinates from the 1st image
        feature_coords_2 (numpy array): coordinates from the 2nd image
        homography (numpy array):       homography matrix
    Returns:
        numpy array: distance computation for each row of the input coordinates
    '''
    if feature_coords_1 is None or feature_coords_1.size == 0 or feature_coords_2 is None or feature_coords_2.size == 0:
        return np.array([])

    transformed_coords_1 = apply_homography_transform(
        feature_coords_1, homography
    )

    return np.linalg.norm(transformed_coords_1 - feature_coords_2, axis=1)


def generate_pointlinedistance_vector_epipolar(feature_coords_1: np.array,
                                               feature_coords_2: np.array,
                                               f_matrix: np.array,
                                               mode: str = 'single') -> np.array:
    '''
    Get the point-line epipolar between the features under epipolar transform (row-level correspondence)

    Args:
        feature_coords_1 (numpy array): coordinates from the 1st image
        feature_coords_2 (numpy array): coordinates from the 2nd image
        f_matrix (numpy array):         fundamental matrix 
        mode (str):                     'single' (one way distances) v 'double' (two way distances)
    Returns:
        numpy array: distance computation for each row of the input coordinates
    '''
    if feature_coords_1 is None or feature_coords_1.shape[0] == 0 or feature_coords_2 is None or feature_coords_2.shape[0] == 0:
        return np.array([])

    if feature_coords_1.shape[1] == 2:
        # convert to homogenous coordinates
        feature_coords_1 = convert_to_homogenous(feature_coords_1)

    if feature_coords_2.shape[1] == 2:
        # convert to homogenous coordinates
        feature_coords_2 = convert_to_homogenous(feature_coords_2)

    epipolar_lines_1 = convert_to_epipolar_lines(feature_coords_1, f_matrix)
    # normalizing the lines for the first two columns
    epipolar_lines_1 = np.divide(epipolar_lines_1, np.linalg.norm(
        epipolar_lines_1[:, :2], axis=1, keepdims=True))

    left_dist = np.abs(
        np.sum(np.multiply(epipolar_lines_1, feature_coords_2), axis=1)
    )

    if mode == 'double':
        right_dist = generate_pointlinedistance_vector_epipolar(
            feature_coords_2,
            feature_coords_1,
            np.transpose(f_matrix),
            mode='single'
        )

        return 0.5*(left_dist+right_dist)
    else:
        return left_dist


def generate_distance_matrix_homography(feature_coords_1: np.array,
                                        feature_coords_2: np.array,
                                        homography: np.array) -> np.array:
    '''
    Get the distance between the features under homography transform (all pairs)

    Args:
        feature_coords_1 (numpy array): coordinates from the 1st image
        feature_coords_2 (numpy array): coordinates from the 2nd image
        homography (numpy array):       homography matrix
    Returns:
        numpy array: distance computation for all pairs of the input coordinates
    '''
    if feature_coords_1 is None or feature_coords_1.shape[0] == 0 or feature_coords_2 is None or feature_coords_2.shape[0] == 0:
        return np.array([])

    transformed_coords_1 = apply_homography_transform(
        feature_coords_1, homography)

    distance_matrix = pairwise_distances(
        transformed_coords_1, feature_coords_2)

    return distance_matrix


def generate_pointlinedistance_matrix_epipolar(feature_coords_1: np.array,
                                               feature_coords_2: np.array,
                                               f_matrix: np.array,
                                               mode: str = 'single'):
    '''
    Get the point-line epipolar between the features under epipolar transform (all pairs)

    Args:
        feature_coords_1 (numpy array): coordinates from the 1st image
        feature_coords_2 (numpy array): coordinates from the 2nd image
        f_matrix (numpy array):         fundamental matrix
        mode (str):                     'single' (one way distances) v 'double' (two way distances)
    Returns:
        numpy array: distance computation for all pairs of the input coordinates
    '''

    # handling the case of empty input(s)
    if feature_coords_1 is None or feature_coords_1.size == 0 or feature_coords_2 is None or feature_coords_2.size == 0:
        return np.array([])

    epipolar_lines_1 = convert_to_epipolar_lines(feature_coords_1, f_matrix)

    # normalizing the lines for the first two columns
    epipolar_lines_1 = np.divide(epipolar_lines_1, np.linalg.norm(
        epipolar_lines_1[:, :2], axis=1, keepdims=True))

    # find point to line distance with feature_coords_2 as points using dot products
    distance_matrix = np.abs(pairwise_kernels(
        epipolar_lines_1, convert_to_homogenous(feature_coords_2), metric='linear')
    )

    if mode == 'double':
        distance_matrix = 0.5*(distance_matrix + generate_pointlinedistance_matrix_epipolar(
            feature_coords_2, feature_coords_1,
            np.transpose(f_matrix), mode='single'
        ).T)

    return distance_matrix


def find_closest_match_greedy(distance_matrix: np.array,
                              dist_threshold: float) -> (np.array, np.array):
    '''
    Find the matches using one-to-one greedy selection of distances which are below the distance threshold
    Args:
        distance_matrix (numpy array): the matrix containing distances between every pairs
        dist_threshold (float)    : max distance value to be considered a valid match
    Returns:
        numpy array: coordinates of matches. shape=Nx2
        numpy array: distance for the matches. shape=Nx1
    '''

    if distance_matrix is None or distance_matrix.size == 0:
        return np.array([]), np.array([])

    is_available_1 = np.full((np.shape(distance_matrix)[0],), True)
    is_available_2 = np.full((np.shape(distance_matrix)[1],), True)

    sorted_indices = np.transpose(np.vstack(np.unravel_index(
        np.argsort(np.ravel(distance_matrix)),
        distance_matrix.shape
    )))

    match_idxes = []
    match_distances = []

    for row_idx in range(sorted_indices.shape[0]):
        idx1 = sorted_indices[row_idx, 0]
        idx2 = sorted_indices[row_idx, 1]

        if dist_threshold is not None and distance_matrix[idx1, idx2] > dist_threshold:
            break

        if is_available_1[idx1] and is_available_2[idx2]:
            is_available_1[idx1] = False
            is_available_2[idx2] = False

            match_idxes.append([idx1, idx2])
            match_distances.append(distance_matrix[idx1, idx2])

    return np.array(match_idxes), np.array(match_distances)


def get_coverage(coordinates: np.array,
                 im_size: List[int],
                 disc_radius: int = 25) -> int:
    '''
    Get the area of image covered by the coordinates

    Note: for a single pixel disk, use disc_radius=0

    Args:
        coordinates (numpy array): coordinates on the image
        im_size (list of int): size of the image
        disc_radius (int): radius of disk for each individual coordinate
    Returns:
        int: the number of pixels (area) covered by the coordinates
    '''

    if coordinates.shape == 0:
        return 0

    # discretize the 2d coordinates to pixel coordinates
    discretized_coordinates = np.rint(coordinates[:, :2]).astype(np.uint16)

    # validate the coordinates which are going out of bounds
    is_valid = validate_coordinates(discretized_coordinates, im_size)
    discretized_coordinates = discretized_coordinates[is_valid]

    # NOTE: the code below has been copied from https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/dilation_utils.py#L19
    # ADD PROPER CREDITS
    mask_diff = np.ones((im_size[0], im_size[1])).astype(np.uint8)
    mask_diff[discretized_coordinates[:, 1], discretized_coordinates[:, 0]] = 0
    distance_mask = cv.distanceTransform(
        mask_diff, distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_PRECISE)
    distance_mask = distance_mask.astype(np.float32)

    return np.count_nonzero(distance_mask <= disc_radius, axis=None)


def generate_features_grid(im1_size: List[int],
                           im2_size: List[int],
                           grid_step_size: float = 0.01) -> (np.array, np.array):
    '''
    Generates a grid of points over the 2 images

    Code from: https://github.com/intel-isl/DFE/blob/2781a3eadc35ee17f0f910b8590d72c48ee5da54/dfe/datasets/fundamental_matrix_dataset.py#L52

    Ref: http://vladlen.info/papers/deep-fundamental-supplement.pdf

    Args:
        im1_size (List of int): size of 1st image
        im2_size (List of int): size of 2nd image
        grid_step_size (float): step size between points. normalized by the image shape
    Returns:
        numpy array: grid for 1st image
        numpy array: grid for 2nd image
    '''

    grid_x, grid_y = np.meshgrid(
        np.arange(0, 1, step=grid_step_size), np.arange(
            0, 1, step=grid_step_size)
    )

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    pts1_grid = np.hstack((im1_size[1]*grid_x, im1_size[0]*grid_y))
    pts2_grid = np.hstack((im2_size[1]*grid_x, im2_size[0]*grid_y))

    return pts1_grid, pts2_grid


def filter_by_kpt_response(max_kpts: int,
                           kpts: List[cv.KeyPoint],
                           descriptors: np.array = None) -> (np.array, np.array):
    '''
    Filter opencv's keypoints to just select the top keypoints

    # TODO: write unit test

    Args:
        max_int (int): max keypoints to select
        kpts (opencv keypoints): keypoints to filter
        descriptors (numpy array): the descriptors corresponding to the keypoint
    Returns:
        numpy array: coordinates of top keypoints by the response
        numpy array: corresponding descriptors
    '''
    # from Alex's feature_utils
    features = np.array([[f.pt[0], f.pt[1], f.size] for f in kpts])
    responses = np.array([f.response for f in kpts])

    nb_kpts = max_kpts

    num_features = np.shape(features)[0]
    if num_features < max_kpts:
        nb_kpts = num_features

    order = responses.argsort()[::-1][:nb_kpts]
    features = features[order, :]

    if descriptors is not None:
        descriptors = descriptors[order, :]
        return features, descriptors

    return features


def convert_to_opencv_keypoints(features: np.array) -> List[cv.KeyPoint]:
    '''
    Converts the features from numpy to cv keypoints

    # TODO: write unit test

    Args:
        features (numpy array): 2d numpy array as features. The first dimension is N and second dimension needs to be atleast 2
    Returns:
        list of opencv keypoints
    '''
    # input features is a 2D array
    if features.shape[1] < 3:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=2) for f in features]
    else:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=f[2]) for f in features]

    return keypoints


def convert_to_numpy_array(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """
    Converts the cv keypoints to a numpy array, the standard feature
        representation in GTSFM
    Args:
        keypoints (List[cv.KeyPoint]): keypoints representation of the given
            features
    Returns:
        np.ndarray: features
    """

    feat_list = [[kp.pt[0], kp.pt[1], kp.size, kp.response]
                 for kp in keypoints]

    return np.array(feat_list, dtype=np.float32)


def check_covisible_homography(coordinates: np.array,
                               homography: np.array,
                               im2_size: List[int]
                               ) -> np.array:
    '''
    Checks if the coordinates will be inside the image frame after the homography transform

    Args:
        coordinates (numpy array):  image coordinates from 1st image
        homography (numpy array):   homography matrix from 1st image to 2nd image
        im2_size (list of int):     size of 2nd image as list
    Returns:
        boolean numpy array: the boolean flag indicating if the point lies inside the image after the homography
    '''
    if coordinates is None or coordinates.size == 0:
        return np.array([])

    transformed_coordinates = apply_homography_transform(
        coordinates[:, :2], homography
    )

    return validate_coordinates(transformed_coordinates, im2_size[:2])


def keypoints_of_array(features: np.ndarray) -> List[cv.KeyPoint]:
    """
    Converts the features from numpy array to cv keypoints.

    Args:
        features (np.ndarray): features as numpy array

    Returns:
        List[cv.KeyPoint]: keypoints representation of the given features
    """
    # TODO(ayush): Make this primary function
    # TODO(ayush): what should be scale if not provided?

    # input features is a 2D array
    if features.shape[1] < 3:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=2) for f in features]
    else:
        keypoints = [cv.KeyPoint(x=f[0], y=f[1], _size=f[2]) for f in features]

    return keypoints


def array_of_keypoints(keypoints: List[cv.KeyPoint]) -> np.ndarray:
    """
    Converts the cv keypoints to a numpy array, the standard feature representation in GTSFM

    Args:
        keypoints (List[cv.KeyPoint]): keypoints representation of the given features

    Returns:
        np.ndarray: features
    """
    # TODO(ayush): Make this primary function

    if len(keypoints) == 0:
        return np.array([])

    response_scores = [kp.response for kp in keypoints]
    # adding an offset to prevent division by zero
    max_response = max(response_scores) + 1e-6
    min_response = min(response_scores)

    feat_list = [[kp.pt[0], kp.pt[1], kp.size, (kp.response-min_response)/(max_response-min_response)]
                 for kp in keypoints]

    return np.array(feat_list, dtype=np.float32)
