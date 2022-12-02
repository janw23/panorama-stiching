from examples import *

from panorama.stich import stich_images_automatically

stiched = None

counter = 0
while stiched is None:
    # core of task7
    stiched = stich_images_automatically(calib2, calib1)
    counter += 1

if counter == 1:
    print('Stiched images on first try')
else:
    print(f'Stiched images after {counter} tries')

cv2.imshow('stiched', stiched)
cv2.waitKey(0)

