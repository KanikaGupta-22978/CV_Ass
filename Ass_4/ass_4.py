# -*- coding: utf-8 -*-
"""ass_4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UN_0RMFIXiqnhoHymUn1CjMe79NInDhE
"""

import cv2
import numpy as np

# World coordinates (3D points)
WorldPoints = np.array([
    [10.9, 10.7, 42],
    [5.5, 3.9, 46.8],
    [14.2, 3.9, 47.0],
    [22.8, 3.9, 47.4],
    [5.5, 10.6, 44.2],
    [14.2, 10.6, 43.8],
    [22.8, 10.6, 44.8],
    [5.5, 17.3, 43],
    [14.2, 17.3, 42.5],
    [22.8, 17.3, 44.4]
])

# Image coordinates (2D points)
ImagePoints = np.array([
    [6.28, 3.42],
    [502, 185],
    [700, 197],
    [894, 208],
    [491, 331],
    [695, 342],
    [896, 353],
    [478, 487],
    [691, 497],
    [900, 508]
])

# Reshaping for the cv2.calibrateCamera function
object_points = np.array([WorldPoints], dtype=np.float32)
image_points = np.array([ImagePoints], dtype=np.float32)

# Camera matrix (estimated initial values)
# Assume fx and fy are roughly equal to image width/height
# Assume principal point is at the center of the image
camera_matrix = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)

# Distortion coefficients (assumed to be zero for simplicity)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Perform camera calibration to get intrinsic and extrinsic parameters
# flags=cv2.CALIB_USE_INTRINSIC_GUESS tells OpenCV to use your camera_matrix as a starting point
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (640, 480), camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

# Output intrinsic parameters
print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("Distortion Coefficients:\n", dist)

# Extrinsic Parameters
print("\nRotation Vectors (Extrinsic Parameters - Rotation):\n", rvecs)
print("\nTranslation Vectors (Extrinsic Parameters - Translation):\n", tvecs)
