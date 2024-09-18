# -*- coding: utf-8 -*-
"""ass_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BEaoK43ewp0wYbDYnlypTjuKl19bH4cA
"""

# Question3-Implementing hybrid image generation using spatial and frequency domain technique
import cv2   # tool for image processing
import matplotlib.pyplot as plt    # tool for data visaulization
import numpy as np       # tool to operate these arrays

# Spatial Domain-
# combining 2 images to create a hybrid image using spatial filtering technique
img1=cv2.imread("/content/dog.jpg", cv2.IMREAD_COLOR)    # read an image
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)    # converted bgr to rgb for display

img2=cv2.imread("/content/cat.jpg", cv2.IMREAD_COLOR)    # same for image 2
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

low_pass_img1=cv2.GaussianBlur(img1, (17,17), 7)      # Blurring image 1
# Parameters: image, kernel size, sigma

plt.imshow(low_pass_img1)

blurred_img2=cv2.GaussianBlur(img2, (17,17), 7)      # same for image 2
high_pass_img2=cv2.subtract(img2, blurred_img2)            # subtracting original from blurred image 2

plt.imshow(high_pass_img2)

hybrid_image=cv2.add(low_pass_img1, high_pass_img2)      # adding low pass image 1 with high pass image 2

plt.imshow(hybrid_image)

# increase in kernal sizes and sigma lead to increase in blurring effect.

# 1. Gaussian Pyramid
# A Gaussian Pyramid is a series of images where each level is a progressively smaller (and blurred) version of the original image.

# 2. Laplacian Pyramid
# A Laplacian Pyramid is derived from the Gaussian Pyramid and represents the difference between successive levels of the Gaussian Pyramid.

# To generate Gaussian Pyramids
def generate_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

# To generate Laplacian Pyramids
def generate_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        if gaussian_expanded.shape != gaussian_pyramid[i].shape:
            gaussian_expanded = cv2.resize(gaussian_expanded, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Last level is the same
    return laplacian_pyramid

# Blend (merging) images using the Gaussian and Laplacian Pyramids
def blend_images_with_pyramids(img1, img2, levels):
    # Generate Gaussian pyramids
    gaussian_pyramid1 = generate_gaussian_pyramid(img1, levels)
    gaussian_pyramid2 = generate_gaussian_pyramid(img2, levels)

    # Generate Laplacian pyramids
    laplacian_pyramid1 = generate_laplacian_pyramid(gaussian_pyramid1)
    laplacian_pyramid2 = generate_laplacian_pyramid(gaussian_pyramid2)

    # Blend the Laplacian pyramids
    blended_pyramid = [cv2.addWeighted(lap1, 0.5, lap2, 0.5, 0) for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2)]

    # Reconstruct the blended image
    blended_image = blended_pyramid[-1]
    for lap in reversed(blended_pyramid[:-1]):
        blended_image = cv2.pyrUp(blended_image)
        if blended_image.shape != lap.shape:
            blended_image = cv2.resize(blended_image, (lap.shape[1], lap.shape[0]))
        blended_image = cv2.add(blended_image, lap)

    return blended_image

blended_image=blend_images_with_pyramids(img1, img2, levels=4)
plt.imshow(blended_image);

# Bilateral Filters are used to preserve the edges while smoothing the image, whoch makes them useful for blending an image.
def bilateral_filter(image, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def blend_images_with_pyramids(img1, img2, levels, bilateral_params):
    # Generate Gaussian pyramids
    gaussian_pyramid1 = generate_gaussian_pyramid(img1, levels)
    gaussian_pyramid2 = generate_gaussian_pyramid(img2, levels)

    # Apply bilateral filter to each level of Gaussian pyramids
    bilateral_gaussian_pyramid1 = [bilateral_filter(level, *bilateral_params) for level in gaussian_pyramid1]
    bilateral_gaussian_pyramid2 = [bilateral_filter(level, *bilateral_params) for level in gaussian_pyramid2]

    # Generate Laplacian pyramids from filtered Gaussian pyramids
    laplacian_pyramid1 = generate_laplacian_pyramid(bilateral_gaussian_pyramid1)
    laplacian_pyramid2 = generate_laplacian_pyramid(bilateral_gaussian_pyramid2)

    # Blend the Laplacian pyramids
    blended_pyramid = [cv2.addWeighted(lap1, 0.5, lap2, 0.5, 0) for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2)]

    # Reconstruct the blended image
    blended_image_2 = blended_pyramid[-1]
    for lap in reversed(blended_pyramid[:-1]):
        blended_image_2 = cv2.pyrUp(blended_image)
        if blended_image_2.shape != lap.shape:
            blended_image_2 = cv2.resize(blended_image_2, (lap.shape[1], lap.shape[0]))
        blended_image_2 = cv2.add(blended_image_2, lap)

    return blended_image_2

blended_image_2=blend_images_with_pyramids(img1, img2, levels=5, bilateral_params=(5, 67, 69))
plt.imshow(blended_image_2)

# Frequency Domain-This method uses Fourier Transforms to manipulate images in the
# frequency domain before blending them and then converting them back to the spatial domain.