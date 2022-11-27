import numpy as np

def compute_blending_weigths(image: np.ndarray) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    xs, ys = xs.astype(float), ys.astype(float)
    
    # Compute weigths based on distance from image edges.
    weights = np.minimum(np.minimum(xs, image.shape[0] - xs), np.minimum(ys, image.shape[1] - ys))

    # Normalize to [0, 1] range.
    weights -= weights.min()
    weights /= weights.max()
    return weights.T

if __name__ == '__main__':
    import cv2
    
    img = cv2.imread('undistorted.png')
    weights = compute_blending_weigths(img[:300])
    print(weights.min(), weights.max())

    cv2.imshow('weights', weights)
    cv2.waitKey(0)
