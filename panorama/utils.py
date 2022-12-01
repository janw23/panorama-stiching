import numpy as np


def compute_bounds_and_offset(xs: np.ndarray, ys: np.ndarray):
    ''' Computes top-left coordinates and size of a bounding box
        encapsulating provided points. '''
    tl = (xs.min(), ys.min())
    br = (xs.max(), ys.max())
    bounds = (br[0] - tl[0], br[1] - tl[1])
    return bounds, tl


def points_to_rays(points: np.ndarray) -> np.ndarray:
    assert points.shape[0] == 2
    return np.vstack([points, np.ones(points.shape[1])])


def rays_to_points(rays: np.ndarray) -> np.ndarray:
    assert rays.shape[0] == 3
    return rays[:2, :] / rays[2, :]


def swap_coordinates(points: np.ndarray):
    ''' Swaps indexing order of points between matrix-like and image-like. '''
    assert points.shape[0] == 2
    return points[::-1, :]


def apply_homography(homography: np.ndarray, points: np.ndarray):
    ''' Applies image-like homography transformation to points expressed in matrix-like coordinates. '''
    return swap_coordinates(rays_to_points(homography @ points_to_rays(swap_coordinates(points))))
