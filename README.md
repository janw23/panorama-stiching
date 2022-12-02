Author: `Jan Wawszczak`

# Project structure
 - `panorama` - core code of the assignment
 - `calibration` - camera parameters, raw and undistorted images
 - `examples` - usage examples based on assignment's tasks. **They are meant to highlight where each task's core implementation is located in the project.**

# Usage
Prepare virtual env
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Stich two images automatically
```
python -m panorama stich <image1_path> <image2_path> -o <output_path>
```
Undistort image using provided camera parameters
```
python -m panorama undistort -i <image_path> -m <intrinsic_path> -d <dist_coeffs_path> -o <output_path>
```

Run examples
```
python -m examples.task<1-7>
```

# Stiched image example
<img src=stiched.png></img>