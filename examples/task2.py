from examples import *

from panorama.transform import transform_image

homography = np.array(
    [[1, 0.1, 0],
     [-0.5, 1.2, 0],
     [0, 0, 1]],
    dtype=float
)

image = calib2

# core of task2
trans, _ = transform_image(image, homography)

scale = image.shape[0] / trans.shape[0]
trans = cv2.resize(trans, (round(trans.shape[1] * scale), image.shape[0]))

side_by_side = cv2.hconcat([image, trans])

cv2.imshow('original and transformed', side_by_side)
cv2.waitKey(0)

