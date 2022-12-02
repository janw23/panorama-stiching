from examples import *
from panorama.camera import CameraInfo

raw = raw2

# core of task1
cam = CameraInfo(intrinsic, dist_coeffs, (raw.shape[1], raw.shape[0]))
undistorted = cam.undistort_image(raw)

# Images are *almost* the same because camera has almost no distortion.
print('raw == undistorted:', np.allclose(raw, undistorted))

side_by_side = cv2.hconcat([raw, undistorted])
cv2.imshow('raw and undistorted', side_by_side)
cv2.waitKey(0)