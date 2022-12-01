from camera_package.camera import Camera

from panorama.stich import stich_images_automatically

import cv2

if __name__ == '__main__':
    cam = Camera(quality=12)
    reference_frame = None

    while True:
        cam.keep_stream_alive()
        frame = cam.get_frame()
        assert frame.shape == (1024, 1280, 3)

        scale_div = 5
        frame = cv2.resize(
            frame, (frame.shape[1] // scale_div, frame.shape[0] // scale_div))

        if reference_frame is None:
            # frame = cv2.resize(frame, (frame.shape[1] * scale_div, frame.shape[0] * scale_div))
            cv2.imshow('frame', frame)
        else:
            stiched = stich_images_automatically(frame, reference_frame)
            if stiched is not None:
                # frame = cv2.resize(frame, (frame.shape[1] * scale_div, frame.shape[0] * scale_div))
                cv2.imshow('frame', stiched)
            else:
                print('couldn\'t stich')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            reference_frame = frame.copy()
