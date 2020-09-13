'''
From Alex's feature_utils.py

'''

import numpy as np
import cv2 as cv

from utils.features import apply_homography_transform, validate_coordinates


def bgr_to_gray_matlab(image):
    """
    Convert image to gray image (Matlab coeffients).

    :param image: The image
    :type image: array
    :returns: gray_image
    :rtype: array(w*h)
    """

    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = np.average(image[:, :, :3], weights=[
                           0.1140, 0.5870, 0.2989], axis=2)
    elif image.shape[2] == 3:
        image = np.average(image[:, :, :3], weights=[
                           0.1140, 0.5870, 0.2989], axis=2)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image


def rgb_to_gray_matlab(image):
    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = (np.average(image[:, :, :3], weights=[
            0.2989, 0.5870, 0.1140], axis=2)).astype(np.uint8)

    elif image.shape[2] == 3:
        image = (np.average(image[:, :, :3], weights=[
            0.2989, 0.5870, 0.1140], axis=2)).astype(np.uint8)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image


def bgr_to_gray_cv(image):
    """
    Convert image to gray image (opencv coeffients).

    :param image: The image
    :type image: array
    :returns: gray_image
    :rtype: array(w*h)
    """

    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    elif image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image


def rgb_to_gray_cv(image):
    """
    Convert image to gray image (opencv coeffients).

    :param image: The image
    :type image: array
    :returns: gray_image
    :rtype: array(w*h)
    """

    if len(image.shape) == 2:
        pass
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image


def convert_to_3channel(image):
    """
    Convert image to 3-channel image.

    :param image: The image
    :type image: array
    :returns: color_image
    :rtype: array(w*h*3)
    """

    if image.ndim == 2:
        image = np.expand_dims(image, 2)
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    else:
        raise ValueError('Input image dimensions are wrong')

    return image


def rectify_patch(img, kp, patch_sz=32):
    """
    Extract an rectified patch from image with information in the keypoint.

    :param img: The image
    :type img: array
    :param kp: The key point
    :type kp: array
    :param patch_sz: patch size
    :type patch_sz: int
    :returns: patch
    :rtype: array(w*h)
    """

    scale = 1.0  # rotate in the patch
    M = cv.getRotationMatrix2D(
        (patch_sz / 2, patch_sz / 2), -1 * kp[3] * 180 / 3.1415, scale)
    # print(M)
    patch = cv.warpAffine(img, np.float32(M), (patch_sz, patch_sz),
                          flags=cv.WARP_INVERSE_MAP + cv.INTER_CUBIC)
    return patch


def extract_patch(img, feature_coordinate, patch_sz=32, rectify_flag=False):
    """
    Extract an rectified patch from image with information in the keypoint.

    :param img: The image
    :type img: array
    :param kp: The key point
    :type kp: array
    :param patch_sz: patch size
    :type patch_sz: int
    :param rectify_flag: rectified or not
    :type rectify_flag: boolean
    :returns: patch
    :rtype: array(w*h)
    """
    img = np.float32(img)

    if feature_coordinate.ndim > 2:
        sub = cv.getRectSubPix(img, (int(feature_coordinate[2] / 2 * patch_sz),
                                     int(feature_coordinate[2] / 2 * patch_sz)), (feature_coordinate[0], feature_coordinate[1]))
        res = cv.resize(sub, (patch_sz, patch_sz))
    else:
        res = cv.getRectSubPix(img, (patch_sz, patch_sz),
                               (feature_coordinate[0], feature_coordinate[1]))

    if rectify_flag:
        res = rectify_patch(res, feature_coordinate, patch_sz)
    return np.asarray(res, dtype=np.uint8)


def image_resize(image_array, max_len=640, inter=cv.INTER_AREA):
    """ returns resized image and inverse scale factor"""
    dim = None
    h, w = image_array.shape[:2]

    s = 1
    if h > w:
        s = max_len/float(h)
    else:
        s = max_len/float(w)

    dim = (int(h * s), int(w * s))

    resized = cv.resize(image_array, dim, interpolation=inter)
    inv_scale = 1/float(s)
    # return the resized image
    return resized, inv_scale


def compute_covisible_image_fraction_homography(im1_size, im2_size, homography):
    '''
    Computes co-visiblity in a single direction (im1's covisiblity)
    '''
    if im1_size is None or len(im1_size) < 2 or im2_size is None or len(im2_size) < 2:
        return 0

    # create a meshgrid on image 1
    x_linear = np.linspace(0, im1_size[1], 100)
    y_linear = np.linspace(0, im1_size[0], 100)

    x_grid, y_grid = np.meshgrid(x_linear, y_linear)

    x_grid = x_grid.flatten().reshape(-1, 1)
    y_grid = y_grid.flatten().reshape(-1, 1)

    grid_coordinates = np.hstack((x_grid, y_grid))

    # apply the homography transform to the grid coordinates
    transformed_coordinates = apply_homography_transform(
        grid_coordinates, homography)

    num_valid_coordinates = np.sum(
        validate_coordinates(transformed_coordinates, im2_size),
        axis=None
    )

    num_total_coordinates = np.shape(x_grid)[0]

    return num_valid_coordinates/num_total_coordinates
