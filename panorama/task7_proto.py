from typing import Union

from .task3_proto import find_homography, rays_to_points, points_to_rays
from .task6_proto import get_matches

import numpy as np

# I want to find homography based on matches with outliers present.
# loop:
#   data_subset = sample(data)
#   model = fit(data_subset)
#   inliers = get_inliers(model, data)
#   if len(inliers) > best_so_far:
#       best_model = fit(inliers)

def find_homography_ransac(source_points: np.ndarray, target_points: np.ndarray, num_iterations: int, threshold: float) -> Union[np.ndarray, None]:
    assert source_points.shape[0] == 2
    assert source_points.shape == target_points.shape

    best_inliers_count = 0
    best_inliers = None

    for _ in range(num_iterations):
        # sample
        min_data_points = 6
        sample_indices = np.random.choice(np.arange(source_points.shape[1]), min_data_points)
        source_subset = source_points[:, sample_indices]
        target_subset = target_points[:, sample_indices]

        # fit
        homography = find_homography(source_subset, target_subset)

        # check inliers
        trans_points = rays_to_points(homography @ points_to_rays(source_points))
        sqr_dists = ((trans_points - target_points) ** 2).sum(axis=0) # compute sqr distance between each target and transformed point
        inliers_mask = sqr_dists <= threshold
        inliers_count = inliers_mask.sum()

        # update inliers that should produce the best model
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers = np.copy(inliers_mask)

    if best_inliers is None:
        return None
    
    homography = find_homography(source_points[:, best_inliers], target_points[:, best_inliers])
    return homography

if __name__ == '__main__':
    import cv2

    img1_fp = 'mc1.png'
    img2_fp = 'mc2.png'

    matches = get_matches(img1_fp, img2_fp, visualize=False)
    img1_pts, img2_pts = list(zip(*matches))
    img1_pts, img2_pts = np.array(img1_pts).T, np.array(img2_pts).T

    from task3_proto import find_homography
    from task4_proto import stich_images

    homography = find_homography_ransac(img1_pts, img2_pts, 100, 25)
    assert homography is not None

    img1 = cv2.imread(img1_fp)
    img2 = cv2.imread(img2_fp)

    stiched = stich_images(img1, img2, homography)

    cv2.imshow('stiched', stiched)
    cv2.waitKey(0)
