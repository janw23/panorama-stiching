from task3_proto import find_homography
from task2_proto import transform_image, compute_bounds_and_offset

import numpy as np
import cv2


def stich_images(adjustable_image: np.ndarray, fixed_image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ''' Stich images into one image by applying homography to adjustable_image and blending the result with fixed_image. '''
    transformed_image, transformed_image_offset = transform_image(adjustable_image, homography)

    # Swap axes ordering so that we can work with images just like with matrices.
    np.swapaxes(adjustable_image, 0, 1)
    np.swapaxes(fixed_image, 0, 1)

    # Compute new image shape and its origin offset w.r.t. fixed_image's origin.

    transformed_xs = np.array([0, transformed_image.shape[0]]) + transformed_image_offset[0]
    transformed_ys = np.array([0, transformed_image.shape[1]]) + transformed_image_offset[1]
    fixed_xs = np.array([0, fixed_image.shape[0]])
    fixed_ys = np.array([0, fixed_image.shape[1]])

    stiched_image_shape = list(fixed_image.shape)
    stiched_image_shape[:2], stiched_image_offset = compute_bounds_and_offset(
        np.hstack([fixed_xs, transformed_xs]), np.hstack([fixed_ys, transformed_ys]))
    stiched_image_shape = tuple(stiched_image_shape)

    # Copy images onto the new stiched image in their correct positions.

    stiched_image_offset = np.array(stiched_image_offset)
    transformed_image_offset = np.array(transformed_image_offset)
    stiched_image = np.zeros(stiched_image_shape, fixed_image.dtype)
    
    tl = -stiched_image_offset + transformed_image_offset
    br = (tl[0] + transformed_image.shape[0], tl[1] + transformed_image.shape[1])
    stiched_image[tl[0]:br[0], tl[1]:br[1]] = transformed_image

    tl = -stiched_image_offset
    br = (tl[0] + fixed_image.shape[0], tl[1] + fixed_image.shape[1])
    stiched_image[tl[0]:br[0], tl[1]:br[1]] = fixed_image

    # Transform image back into image-like coordinates.
    np.swapaxes(stiched_image, 0, 1)
    return stiched_image


img1 = cv2.imread('../lab3/captures_esp32/frame_10.png')
img2 = cv2.imread('../lab3/captures_esp32/frame_11.png')

img1_points = [
    [194, 172],
    [163, 517],
    [748, 645],
    [223, 834],
    [768, 331],
]

img2_points = [
    [543, 166],
    [524, 496],
    [1117, 612],
    [611, 803],
    [1133, 267],
]

img1_points = np.array(img1_points).T
img2_points = np.array(img2_points).T

homography = find_homography(img2_points, img1_points)

stiched_image = stich_images(img2, img1, homography)
cv2.imshow('stiched', stiched_image)
cv2.waitKey(0)

exit()


img, origin_offset = transform_image(img2, homography)

cv2.imshow('before', img2)
cv2.imshow('after', img)
cv2.waitKey(0)

# concated = cv2.hconcat([img2, img])
# cv2.imshow('before and after', concated)
# cv2.waitKey(0)

img_draw = img1.copy()
nonzero = img != 0
img_draw[nonzero] = img[nonzero]
cv2.imshow('overlapped', img_draw)
cv2.waitKey(0)
