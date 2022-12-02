from pathlib import Path
import numpy as np
import cv2

calib_dir = Path(__file__).parent.parent / 'calibration'
intrinsic = np.loadtxt(calib_dir / 'intrinsic.gz')
dist_coeffs = np.loadtxt(calib_dir / 'dist_coeffs.gz')

raw1 = cv2.imread(str(calib_dir / 'raw1.png'))
raw2 = cv2.imread(str(calib_dir / 'raw2.png'))
calib1 = cv2.imread(str(calib_dir / 'calib1.png'))
calib2 = cv2.imread(str(calib_dir / 'calib2.png'))