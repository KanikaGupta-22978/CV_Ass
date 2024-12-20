# -*- coding: utf-8 -*-
"""Ass_6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Dyc1PRmBJO4zXeIRdQ7Co9jXBAmAidYS
"""

# Automatic method
import cv2
import matplotlib.pyplot as plt

# Function to display images in a grid
def show_images_grid(image1, image2, stitched):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Display the two input images on the top row
    ax[0].imshow(cv2.cvtColor(cv2.hconcat([image1, image2]), cv2.COLOR_BGR2RGB))
    ax[0].set_title("Input Images")
    ax[0].axis("off")

    # Display the stitched image on the second row
    ax[1].imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Stitched Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

# Load images
image1 = cv2.imread(r'/content/k1.jpg')  # Replace with your image path
image2 = cv2.imread(r'/content/k2.jpg')  # Replace with your image path

# List of images to stitch
images = [image1, image2]

# Create a stitcher instance
stitcher = cv2.Stitcher_create()

# Perform stitching
status, stitched_image = stitcher.stitch(images)

# Check the status
if status == cv2.Stitcher_OK:
    print("Stitching completed successfully!")
    # Show images
    show_images_grid(image1, image2, stitched_image)
else:
    print(f"Stitching failed with status code: {status}")