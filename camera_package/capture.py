import cv2
import pathlib
from pathlib import Path
import numpy as np
import frame_builders, camera

cam = camera.Camera(quality=12)
# vid = cv2.VideoCapture('tcp://192.168.0.70:10000')
counter = 0

intrinsic = np.loadtxt('../lab3/esp32_intrinsic.gz').astype('float32')
mapx = np.loadtxt('esp32_mapx.gz').astype('float32')
mapy = np.loadtxt('esp32_mapy.gz').astype('float32')


def coordinates(point):
    return tuple([int(i) for i in tuple(point.ravel())])


def draw(img, corners, imgpts):
    corner = coordinates(corners[0].ravel())
    img = cv2.line(img, corner, coordinates(imgpts[0]), (255, 0, 0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[1]), (0, 255, 0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[2]), (0, 0, 255), 5)
    return img

chess_size = (5, 8)
objpoints = [(0, y, x) for y in range(chess_size[1])
             for x in range(chess_size[0])]
objpoints = np.array(objpoints, dtype=float).reshape(-1, 3)

while True:
    # ret, frame = vid.read()
    cam.keep_stream_alive()
    frame = cam.get_frame()
    assert frame.shape == (1024, 1280, 3)
    
    # dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # img = cv2.resize(dst, (dst.shape[1] // 2, dst.shape[0] // 2))
    # cv2.imshow('frame', img)

    ###################################
    img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    found, corners = cv2.findChessboardCorners(img, chess_size, None)
    if found:
        # TODO subpixel?
        cv2.drawChessboardCorners(img, chess_size, corners, True)
        # use known points to solve pnp
        ret, rvec, tvec = cv2.solvePnP(objpoints, corners, intrinsic, 0)

        assert ret
        # draw axes based on known rvec and rvec
        axes_obj = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axes_obj, rvec, tvec, intrinsic, 0)
        # imgpts = imgpts.reshape(-1, 2)
        # corners = corners.reshape(-1, 2)
        img = draw(img, corners, imgpts)

    cv2.imshow('img', img)
    ##################################



    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord(' '):
        ret = cv2.imwrite(f'captures_esp32/frame_{counter}.png', frame)
        counter += 1
        print(ret)

# vid.release() 
cv2.destroyAllWindows() 