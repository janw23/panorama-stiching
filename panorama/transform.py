from typing import Tuple

import numpy as np

from .utils import apply_homography, compute_bounds_and_offset


def transform_image(image: np.ndarray, homography: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    '''Transforms image using given homography.'''
    assert homography.shape == (3, 3)
    np.swapaxes(image, 0, 1)  # transform image into matrix-like coordinates

    # Find shape of transformed image and offset of its origin w.r.t. original image's origin.
    corners = np.array([
        [0, 0],
        [0, image.shape[1]],
        [image.shape[0], 0],
        [image.shape[0], image.shape[1]]]).T
    corners = apply_homography(homography, corners)
    xs, ys = np.rint(corners).astype(int)
    des_shape = list(image.shape)
    des_shape[:2], des_origin_offset = compute_bounds_and_offset(xs, ys)
    des_shape = tuple(des_shape)

    # Using inverse homography, for each pixel in the transformed image
    # find pixel in the source image that is the source of its color value.

    xs, ys = np.mgrid[:des_shape[0], :des_shape[1]]
    xs += des_origin_offset[0]
    ys += des_origin_offset[1]
    des_coords = np.vstack([xs.ravel(), ys.ravel()])

    homography_inv = np.linalg.inv(homography)
    src_coords = apply_homography(homography_inv, des_coords)
    src_coords = np.rint(src_coords).astype(int)  # nearest neighbour

    # Check which pixels fall outside original image.
    outside_mask = (src_coords[0] < 0) | (src_coords[0] >= image.shape[0]) | (
        src_coords[1] < 0) | (src_coords[1] >= image.shape[1])

    src_coords[:, outside_mask] = 0
    des = image[src_coords[0], src_coords[1]]
    des[outside_mask] = 0

    des = des.reshape(des_shape)
    np.swapaxes(des, 0, 1)  # transform image into image-like coordinates

    return des, des_origin_offset
