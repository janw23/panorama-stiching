from camera_package.camera import Camera

from panorama.task6_proto import get_matches
from panorama.task7_proto import find_homography_ransac
from panorama.task4_proto import stich_images

import cv2
import numpy as np


if __name__ == '__main__':
    cam = Camera(quality=12)
    reference_frame = None

    while True:
        cam.keep_stream_alive()
        frame = cam.get_frame()
        assert frame.shape == (1024, 1280, 3)

        scale_div = 3
        frame = cv2.resize(frame, (frame.shape[1] // scale_div, frame.shape[0] // scale_div))

        if reference_frame is None:
            cv2.imshow('frame', frame)
        else:
            matches = get_matches(frame, reference_frame, visualize=False)
            # TODO instead of doing this everywhere modify functions
            if len(matches) >= 4:
                src_pts, tgt_pts = list(zip(*matches))
                src_pts = np.array(src_pts).T
                tgt_pts = np.array(tgt_pts).T
                homography = find_homography_ransac(src_pts, tgt_pts, 1000, 20)

                if homography is None:
                    print('homography not found')
                else:
                    stiched = stich_images(frame, reference_frame, homography)
                    cv2.imshow('frame', stiched)
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        if key == ord(' '):
            reference_frame = frame.copy()