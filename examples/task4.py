from examples import *

from panorama.homography import find_homography


pts1 = np.array(
    [[16, 618],
    [508, 740],
    [50, 961],
    [134, 695],
    [7, 817]]
).T

pts2 = np.array(
    [[638, 611],
    [1094, 762],
    [657, 923],
    [734, 687],
    [625, 789]]
).T

# core of task4
homography = find_homography(pts2, pts1)

if __name__ == '__main__':
    print(homography)