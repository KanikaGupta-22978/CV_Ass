# Manual Method
import cv2
import numpy as np

# Global variables for mouse click
points_img1 = []
points_img2 = []
current_image = None
current_points = None

def select_points(event, x, y, flags, param):
    """
    Callback function to select points on the image.
    """
    global current_image, current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        current_points.append((x, y))
        cv2.imshow("Select Points", current_image)

def manually_select_points(image, num_points=4):
    """
    Opens an image window for manual point selection.
    """
    global current_image, current_points
    current_image = image.copy()
    current_points = []
    cv2.imshow("Select Points", current_image)
    cv2.setMouseCallback("Select Points", select_points)
    print(f"Please select {num_points} points on the current image.")
    
    while len(current_points) < num_points:
        cv2.waitKey(1)
    
    cv2.destroyWindow("Select Points")
    return np.array(current_points, dtype=np.float32)

def blend_images(img1, img2):
    """
    Blends two images smoothly using multi-band blending.
    Ensures the images have the same dimensions before blending.
    """
    # Resize img2 to match the dimensions of img1
    h1, w1 = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w1, h1))

    # Ensure both images have the same number of channels
    if img1.shape[2] != img2_resized.shape[2]:
        if img1.shape[2] > img2_resized.shape[2]:
            img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
        else:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # Perform weighted blending
    blended = cv2.addWeighted(img1, 0.5, img2_resized, 0.5, 0)
    return blended


def stitch_images(img1, img2, H):
    """
    Stitches two images together using the homography matrix.
    """
    # Warp the first image into the second image's perspective
    height, width, channels = img2.shape
    warped_img1 = cv2.warpPerspective(img1, H, (width * 2, height))

    # Place the second image on the canvas
    stitched = warped_img1.copy()
    stitched[0:height, 0:width] = img2

    return stitched

def main():
    img1 = cv2.imread(r"C:\Users\ftska\Downloads\k1.jpg")
    img2 = cv2.imread(r"C:\Users\ftska\Downloads\k2.jpg")
    
    if img1 is None or img2 is None:
        print("Error: One or both image paths are invalid.")
        return
    
    # Resize images for easier processing (optional)
    img1 = cv2.resize(img1, (800, 600))
    img2 = cv2.resize(img2, (800, 600))

    # Select points manually
    print("Select points on the first image:")
    points_img1 = manually_select_points(img1)
    
    print("Select points on the second image:")
    points_img2 = manually_select_points(img2)

    # Compute homography
    H, status = cv2.findHomography(points_img1, points_img2, method=cv2.RANSAC)
    print(f"Homography Matrix:\n{H}")

    # Stitch the images
    stitched_image = stitch_images(img1, img2, H)

    # Blend the overlapping regions
    blended_image = blend_images(stitched_image, img2)

    # Display and save the stitched image
    cv2.imshow("Stitched Image", blended_image)
    cv2.imwrite("stitched_image_final.jpg", blended_image)
    print("Stitched image saved as 'stitched_image_final.jpg'.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

