# File for prototyping stuff

import numpy as np

# This function implements 'Task 2'
def transform_image(image: np.ndarray, homography: np.ndarray):
    ''' Applies projective transformation to the image.
        Provided homography matrix is assumed to be inversible.'''
    assert homography.shape == (3, 3)

    # des is a destination (transformed) image
    des_shape = image.shape  # TODO compute new shape to fit the entire image

    # Prepare matrix of pixel coordinates to transform using inverse homography.
    xs, ys = np.mgrid[:des_shape[0], :des_shape[1]]
    # 'ys' is first in vstack because axis order in images is reversed w.r.t. standard matrix order
    des_coords = np.vstack(
        [ys.ravel(), xs.ravel(), np.ones(xs.size)]).astype(float)

    # Compute corresponsing pixel coordinates in the source image.
    homography_inv = np.linalg.inv(homography.astype(float))
    print(homography_inv)
    src_coords = homography_inv @ des_coords
    # divide so that z = 1 and extract x, y coords
    # TODO check for division by zero?
    src_coords = src_coords[:2, :] / src_coords[2, :]
    src_coords = np.rint(src_coords).astype(int)  # nearest neighbour

    # Check which pixels fall outside the source image coordinates.
    outside = (src_coords[0] < 0) | (src_coords[0] >= des_shape[1]) | (
        src_coords[1] < 0) | (src_coords[1] >= des_shape[0])

    src_coords[:, outside] = 0
    des = image[src_coords[1], src_coords[0]]
    des[outside] = 0
    des = des.reshape(des_shape)

    concated = cv2.hconcat([image, des])
    cv2.imshow('before and after', concated)
    cv2.waitKey(1)


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
        angle = i / 50
        homography[0, 0] = cos(angle)
        homography[0, 1] = -sin(angle)
        homography[1, 0] = sin(angle)
        homography[1, 1] = cos(angle)
        homography[2, 0] = i / 10000
        transform_image(img, homography)
