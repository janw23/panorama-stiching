from examples import *

from panorama.homography import test_find_homography

# core of task3 - not much to show here actually
if test_find_homography():
    print('Tests passed')
else:
    print('Tests failed')