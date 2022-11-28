from task3_proto import find_homography
from task2_proto import transform_image, compute_bounds_and_offset
from task5_proto import compute_blending_weights

from typing import Tuple

import numpy as np
import cv2

def expand_image(target_size: Tuple[int, int], offset: np.ndarray, image: np.ndarray) -> np.ndarray:
    shape = list(image.shape)
    shape[:2] = target_size
    new_image = np.zeros_like(image, shape=shape)

    tl = offset
    br = (tl[0] + image.shape[0], tl[1] + image.shape[1])
    new_image[tl[0]:br[0], tl[1]:br[1]] = image

    return new_image

def stich_images(adjustable_image: np.ndarray, fixed_image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ''' Stich images into one image by applying homography to adjustable_image and blending the result with fixed_image. '''

    adjustable_image_weights = compute_blending_weights(adjustable_image)
    fixed_image_weights = compute_blending_weights(fixed_image)

    trans_image, trans_image_offset = transform_image(adjustable_image, homography)
    trans_image_weights, _ = transform_image(adjustable_image_weights, homography)

    # Swap axes ordering so that we can work with images just like with matrices.
    np.swapaxes(adjustable_image, 0, 1)
    np.swapaxes(fixed_image, 0, 1)

    # Compute new image shape and its origin offset w.r.t. fixed_image's origin.

    trans_xs = np.array([0, trans_image.shape[0]]) + trans_image_offset[0]
    trans_ys = np.array([0, trans_image.shape[1]]) + trans_image_offset[1]
    fixed_xs = np.array([0, fixed_image.shape[0]])
    fixed_ys = np.array([0, fixed_image.shape[1]])

    stiched_image_shape = list(fixed_image.shape)
    stiched_image_shape[:2], stiched_image_offset = compute_bounds_and_offset(
        np.hstack([fixed_xs, trans_xs]), np.hstack([fixed_ys, trans_ys]))
    stiched_image_shape = tuple(stiched_image_shape)

    # Copy images onto the new stiched image in their correct positions.
    # Offsets are initially relative to fixed_image's origin and we transform them to be
    # relative to stiched_image's origin.

    fixed_image_offset = -np.array(stiched_image_offset)
    trans_image_offset = fixed_image_offset + np.array(trans_image_offset)
    
    # Combine images using weighted average.

    trans_image_full = expand_image(stiched_image_shape[:2], trans_image_offset, trans_image)
    trans_image_weights_full = expand_image(stiched_image_shape[:2], trans_image_offset, trans_image_weights)
    fixed_image_full = expand_image(stiched_image_shape[:2], fixed_image_offset, fixed_image)
    fixed_image_weights_full = expand_image(stiched_image_shape[:2], fixed_image_offset, fixed_image_weights)

    trans_image_weights_full = np.expand_dims(trans_image_weights_full, 2)
    fixed_image_weights_full = np.expand_dims(fixed_image_weights_full, 2)

    # Avoid zero division.
    both_zero = (trans_image_weights_full == 0) & (fixed_image_weights_full == 0)
    trans_image_weights_full[both_zero] = 1
    fixed_image_weights_full[both_zero] = 1

    trans_image_full = trans_image_full.astype(float) * trans_image_weights_full
    fixed_image_full = fixed_image_full.astype(float) * fixed_image_weights_full
    stiched_image = np.rint((trans_image_full + fixed_image_full) / (trans_image_weights_full + fixed_image_weights_full)).astype(fixed_image.dtype)

    # Transform image back into image-like coordinates.
    np.swapaxes(stiched_image, 0, 1)
    return stiched_image

if __name__ == '__main__':

    img1 = cv2.imread('mc1.png')
    img2 = cv2.imread('mc2.png')

    img1_points = [
        (1034, 729),
        (659, 1201),
        (1769, 1252),
        (2293, 1043),
        (1516, 877),
    ]

    img2_points = [
        (541, 518),
        (34, 1127),
        (1331, 979),
        (1694, 776),
        (1117, 674),
    ]

    img1_points = np.array(img1_points).T
    img2_points = np.array(img2_points).T

    homography = find_homography(img2_points, img1_points)
    stiched_image = stich_images(img2, img1, homography)

    cv2.imwrite('stiched.png', stiched_image)

    cv2.imshow('stiched', stiched_image)
    cv2.waitKey(0)