from typing import Union

import numpy as np

from .utils import points_to_rays, rays_to_points


def find_homography(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    ''' Finds homography which transforms one list of points to another.
        Points are assumed to be 2d and in image-like coordinates. '''
    assert len(source_points.shape) == 2
    assert source_points.shape[0] == 2
    assert source_points.shape == target_points.shape

    # Assemble matrix A of optimization problem: ||A @ flattened_homography|| -> min.
    num_points = source_points.shape[1]
    A = np.zeros((2 * num_points, 9), dtype=float)

    rays = points_to_rays(source_points).T
    xx = target_points[0, :] * source_points[0, :]
    xy = target_points[0, :] * source_points[1, :]
    yy = target_points[1, :] * source_points[1, :]
    yx = target_points[1, :] * source_points[0, :]

    even = np.arange(0, A.shape[0], 2)
    odd = np.arange(1, A.shape[0], 2)

    A[even, :3] = rays
    A[even, 6] = -xx
    A[even, 7] = -xy
    A[even, 8] = -target_points[0, :]

    A[odd, 3:6] = rays
    A[odd, 6] = -yx
    A[odd, 7] = -yy
    A[odd, 8] = -target_points[1, :]

    # Solve optimization problem via SVD.
    _, _, V = np.linalg.svd(A)
    homography = V[-1, :]
    return homography.reshape(3, 3)


def find_homography_ransac(source_points: np.ndarray, target_points: np.ndarray, num_iterations: int, threshold: float) -> Union[np.ndarray, None]:
    ''' Finds homography which transforms one list of points to another.
        Points are assumed to be 2d and in image-like coordinates.
        Thanks to RANSAC, this method is more robust to outliers than a simple find_homography() .'''
    assert source_points.shape[0] == 2
    assert source_points.shape == target_points.shape

    min_data_points = 6
    best_inliers_count = 0
    best_inliers = None

    if source_points.shape[1] < min_data_points:
        return None

    for _ in range(num_iterations):
        # sample
        sample_indices = np.random.choice(
            np.arange(source_points.shape[1]), min_data_points)
        source_subset = source_points[:, sample_indices]
        target_subset = target_points[:, sample_indices]

        # fit
        homography = find_homography(source_subset, target_subset)

        # check inliers
        trans_points = rays_to_points(
            homography @ points_to_rays(source_points))
        # compute sqr distance between each target and transformed point
        sqr_dists = ((trans_points - target_points) ** 2).sum(axis=0)
        inliers_mask = sqr_dists <= threshold
        inliers_count = inliers_mask.sum()

        # update inliers that should produce the best model
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers = np.copy(inliers_mask)

    if best_inliers is None or len(best_inliers) < 30:
        return None

    homography = find_homography(
        source_points[:, best_inliers], target_points[:, best_inliers])
    return homography


def test_find_homography() -> bool:
    for t in range(100):
        homography = np.random.randn(3, 3)
        source_points = np.random.randn(2, 100)
        target_points = rays_to_points(homography @ points_to_rays(source_points))

        # Before comparing, normalize homography and make signs equal.
        recovered_homography = find_homography(source_points, target_points)
        homography = homography / np.linalg.norm(homography)
        homography *= np.sign(homography[0, 0]) * \
            np.sign(recovered_homography[0, 0])

        if not np.allclose(homography, recovered_homography):
            print('expected:', homography.ravel())
            print('got:     ', recovered_homography.ravel())
            return False
    return True
