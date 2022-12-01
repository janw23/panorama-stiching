import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['undistort', 'stich'])

# undistort
parser.add_argument('-i', '--image-path', type=Path)
parser.add_argument('-m', '--matrix-path', type=Path)
parser.add_argument('-d', '--distortion-coeffs-path', type=Path)
parser.add_argument('-o', '--output-path', type=Path)

parser.add_argument('img1', type=Path)
parser.add_argument('img2', type=Path)

args = parser.parse_args()

if args.action == 'undistort':
    from camera import CameraInfo
    import numpy as np
    import cv2
    
    image = cv2.imread(str(args.image_path))
    camera_matrix = np.loadtxt(args.matrix_path)
    dist_coeffs = np.loadtxt(args.distortion_coeffs_path)

    camera_info = CameraInfo(camera_matrix, dist_coeffs, (image.shape[1], image.shape[0]))
    image = camera_info.undistort_image(image)
    cv2.imwrite(str(args.output_path), image)
elif args.action == 'stich':
    from .task6_proto import get_matches
    from .task7_proto import find_homography_ransac
    from .task4_proto import stich_images
    import cv2
    import numpy as np

    img1 = cv2.imread(str(args.img1))
    img2 = cv2.imread(str(args.img2))

    matches = get_matches(img1, img2, visualize=False)
    if len(matches) >= 4:
        src_pts, tgt_pts = list(zip(*matches))
        src_pts = np.array(src_pts).T
        tgt_pts = np.array(tgt_pts).T
        homography = find_homography_ransac(src_pts, tgt_pts, 1000, 20)

        if homography is None:
            print('homography not found')
        else:
            stiched = stich_images(img1, img2, homography)
            cv2.imshow('frame', stiched)
            cv2.waitKey(0)
    else:
        print('not enough matches found')



else:
    raise RuntimeError("Unknown action")