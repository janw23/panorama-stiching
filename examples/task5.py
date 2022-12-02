from examples import *
from .task4 import homography

from panorama.stich import stich_images_using_homography

# core of task 5
stiched = stich_images_using_homography(calib2, calib1, homography)

cv2.imshow('stiched', stiched)
cv2.waitKey(0)

