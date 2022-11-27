# File for prototyping stuff
from typing import Union, Tuple

from task3_proto import points_to_rays, rays_to_points

import numpy as np


def swap_coordinates(points: np.ndarray):
    ''' Swaps indexing order of points between matrix-like and image-like. '''
    assert points.shape[0] == 2
    return points[::-1, :]


def apply_homography(homography: np.ndarray, points: np.ndarray):
    ''' Applies homography transformation to points expressed in matrix-like coordinates. '''
    return swap_coordinates(rays_to_points(homography @ points_to_rays(swap_coordinates(points))))

def compute_bounds_and_offset(xs: np.ndarray, ys: np.ndarray):
    tl = (xs.min(), ys.min())
    br = (xs.max(), ys.max())
    bounds = (br[0] - tl[0], br[1] - tl[1])
    return bounds, tl


def transform_image(image: np.ndarray, homography: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
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


if __name__ == '__main__':
    import cv2
    from math import sin, cos

    img = cv2.imread('undistorted.png')
    homography = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        dtype=float
    )

    for i in range(100):
        angle = i / 10
        # homography[0, 1] = i/50
        homography[1, 0] = i/50
        homography[0, 0] = cos(angle)
        homography[0, 1] = -sin(angle)
        homography[1, 0] = sin(angle)
        homography[1, 1] = cos(angle)
        # homography[2, 0] = i / 10000
        # homography[1, 2] = i * 5
        des, _ = transform_image(img, homography)
        cv2.imshow('transformed', des)
        cv2.waitKey(1)
