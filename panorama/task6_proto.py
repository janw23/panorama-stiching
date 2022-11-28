import cv2

# This function is heavily inspired by an OpenCV tutorial:
# https://docs.opencv.org/4.6.0/dc/dc3/tutorial_py_matcher.html
#
# You can use it as is, you do not have to understand the insides.
# You need to pass filenames as arguments.
# You can disable the preview with visualize=False.
# lowe_ratio controls filtering of matches, increasing it
# will increase number of matches, at the cost of their quality.
#
# Return format is a list of matches, where match is a tuple of two keypoints.
# First keypoint designates coordinates on the first image.
# Second one designates the same feature on the second image.
def get_matches(filename1, filename2, visualize=True, lowe_ratio=0.6):
    # Read images from files, convert to greyscale
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

    # Find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    if visualize:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("vis", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return good_matches

if __name__ == '__main__':
    import numpy as np

    img1_fp = 'mc1.png'
    img2_fp = 'mc2.png'

    matches = get_matches(img1_fp, img2_fp, visualize=True)
    img1_pts, img2_pts = list(zip(*matches))
    img1_pts, img2_pts = np.array(img1_pts).T, np.array(img2_pts).T

    from task3_proto import find_homography
    from task4_proto import stich_images

    homography = find_homography(img1_pts, img2_pts)

    img1 = cv2.imread(img1_fp)
    img2 = cv2.imread(img2_fp)

    stiched = stich_images(img1, img2, homography)

    cv2.imshow('stiched', stiched)
    cv2.waitKey(0)
