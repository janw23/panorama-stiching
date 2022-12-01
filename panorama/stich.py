from typing import Tuple, Union

import numpy as np

from .transform import transform_image
from .utils import compute_bounds_and_offset
from .homography import find_homography_ransac
from .match import get_matches


def _compute_blending_weights(image: np.ndarray) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    xs, ys = xs.astype(float), ys.astype(float)

    # Compute weigths based on distance from image edges.
    weights = np.minimum(np.minimum(
        xs + 1, image.shape[0] - xs), np.minimum(ys + 1, image.shape[1] - ys))

    # Normalize to [0, 1] range.
    weights -= weights.min()
    weights /= weights.max()
    return weights.T


def _expand_image(target_size: Tuple[int, int], offset: np.ndarray, image: np.ndarray) -> np.ndarray:
    ''' Pads image with black pixels so that resulting image
        looks like original one translated by the provided offset. '''
    assert offset[0] >= 0 and offset[1] >= 0
    assert target_size[0] >= image.shape[0] + offset[0] and \
        target_size[1] >= image.shape[1] + offset[1]

    shape = list(image.shape)
    shape[:2] = target_size
    new_image = np.zeros_like(image, shape=shape)

    tl = offset
    br = (tl[0] + image.shape[0], tl[1] + image.shape[1])
    new_image[tl[0]:br[0], tl[1]:br[1]] = image

    return new_image


def stich_images_using_homography(adjustable_image: np.ndarray, fixed_image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ''' Stich images into one image by applying homography to adjustable_image and blending the result with fixed_image. '''

    adjustable_image_weights = _compute_blending_weights(adjustable_image)
    fixed_image_weights = _compute_blending_weights(fixed_image)

    trans_image, trans_image_offset = transform_image(
        adjustable_image, homography)
    trans_image_weights, _ = transform_image(
        adjustable_image_weights, homography)

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

    # Offsets are initially relative to fixed_image's origin and we transform them to be
    # relative to stiched_image's origin.

    fixed_image_offset = -np.array(stiched_image_offset)
    trans_image_offset = fixed_image_offset + np.array(trans_image_offset)

    # Combine images using weighted average.

    trans_image_full = _expand_image(
        stiched_image_shape[:2], trans_image_offset, trans_image)
    trans_image_weights_full = _expand_image(
        stiched_image_shape[:2], trans_image_offset, trans_image_weights)
    fixed_image_full = _expand_image(
        stiched_image_shape[:2], fixed_image_offset, fixed_image)
    fixed_image_weights_full = _expand_image(
        stiched_image_shape[:2], fixed_image_offset, fixed_image_weights)

    trans_image_weights_full = np.expand_dims(trans_image_weights_full, 2)
    fixed_image_weights_full = np.expand_dims(fixed_image_weights_full, 2)

    # Avoid zero division.
    both_zero = (trans_image_weights_full == 0) & \
        (fixed_image_weights_full == 0)
    trans_image_weights_full[both_zero] = 1
    fixed_image_weights_full[both_zero] = 1

    trans_image_full = trans_image_full.astype(
        float) * trans_image_weights_full
    fixed_image_full = fixed_image_full.astype(
        float) * fixed_image_weights_full
    stiched_image = np.rint((trans_image_full + fixed_image_full) / (
        trans_image_weights_full + fixed_image_weights_full)).astype(fixed_image.dtype)

    # Transform image back into image-like coordinates.
    np.swapaxes(stiched_image, 0, 1)
    return stiched_image


def stich_images_automatically(adjustable_image: np.ndarray, fixed_image: np.ndarray) -> Union[None, np.ndarray]:
    matches = get_matches(adjustable_image, fixed_image, visualize=False)
    if len(matches) == 0:
        return None

    img1_pts, img2_pts = list(zip(*matches))
    img1_pts, img2_pts = np.array(img1_pts).T, np.array(img2_pts).T

    homography = find_homography_ransac(img1_pts, img2_pts, 1000, 10)
    if homography is None:
        return None

    return stich_images_using_homography(adjustable_image, fixed_image, homography)
