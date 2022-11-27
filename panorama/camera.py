from typing import Union, Tuple
import numpy as np
import cv2


class CameraInfo:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: Union[np.ndarray, int], image_size: Tuple[int, int]):
        """
        @param camera_matrix - intrinsic camera matrix
        @param dist_coeffs - distortion coefficients
        @param image_size - image size in (width, height) order
        """
        assert not (isinstance(dist_coeffs, int) and dist_coeffs == 0)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_size = image_size

        self._undistorted_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, self.image_size, alpha=0)
        self._undistortion_maps = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self._undistorted_camera_matrix, self.image_size, 5)

    def undistorted(self):
        return CameraInfo(self._undistorted_camera_matrix, 0, self.image_size)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        assert (image.shape[1], image.shape[0]) == self.image_size
        return cv2.remap(image, *self._undistortion_maps, interpolation=cv2.INTER_LINEAR)
