import argparse
from pathlib import Path
import sys

assert len(sys.argv) > 2

action = sys.argv[1]
argv = sys.argv[2:]

if action == 'undistort':
    from .camera import CameraInfo
    import numpy as np
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-path', type=Path)
    parser.add_argument('-m', '--matrix-path', type=Path)
    parser.add_argument('-d', '--distortion-coeffs-path', type=Path)
    parser.add_argument('-o', '--output-path', type=Path)
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.image_path))
    camera_matrix = np.loadtxt(args.matrix_path)
    dist_coeffs = np.loadtxt(args.distortion_coeffs_path)

    camera_info = CameraInfo(camera_matrix, dist_coeffs,
                             (image.shape[1], image.shape[0]))
    image = camera_info.undistort_image(image)
    cv2.imwrite(str(args.output_path), image)
    print(f'Saved undistorted image to {args.output_path}')

elif action == 'stich':
    import cv2
    from .stich import stich_images_automatically

    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=Path)
    parser.add_argument('img2', type=Path)
    parser.add_argument('-o', '--output-path', type=Path, required=False)
    args = parser.parse_args(argv)

    img1 = cv2.imread(str(args.img1))
    img2 = cv2.imread(str(args.img2))

    stiched = stich_images_automatically(img1, img2)
    if stiched is None:
        print('Could not stich images')
    else:
        if args.output_path is not None:
            cv2.imwrite(str(args.output_path), stiched)
            print(f'Saved stiched panorama to {args.output_path}')

        cv2.imshow('stiched', stiched)
        cv2.waitKey(0)


else:
    raise RuntimeError("Unknown action")
