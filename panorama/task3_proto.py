import numpy as np


def points_to_rays(points: np.ndarray) -> np.ndarray:
    assert points.shape[0] == 2
    return np.vstack([points, np.ones(points.shape[1])])


def rays_to_points(rays: np.ndarray) -> np.ndarray:
    assert rays.shape[0] == 3
    # TODO check for zero division?
    return rays[:2, :] / rays[2, :]


def find_homography(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    ''' Finds homography which transforms one list of points to another.
        Points are assumed to be 2d, and third dimension with value equal to 1 is appended
        before doing internal computations. '''
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


def test_find_homography() -> bool:
    for t in range(100):
        homography = np.random.randn(3, 3)
        source_points = np.random.randn(2, 100)
        target_points = rays_to_points(
            homography @ points_to_rays(source_points))

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


if __name__ == '__main__':
    print('test passed:', test_find_homography())
