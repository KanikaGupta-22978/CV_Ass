import cv2
import numpy as np
import matplotlib.pyplot as plt

def manually_select_points(img1, img2):
    """
    Select corresponding points manually from two images.
    """
    print("\nStep 1: Manually select corresponding points.")
    
    print("Select points on the first image (Image 1) and press ENTER when done.")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    points1 = plt.ginput(n=-1, timeout=0)  # Unlimited points until ENTER is pressed
    plt.close()

    print("Select corresponding points on the second image (Image 2) and press ENTER when done.")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    points2 = plt.ginput(n=-1, timeout=0)
    plt.close()

    points1 = np.array(points1)
    points2 = np.array(points2)
    print(f"\nSelected {len(points1)} points on Image 1 and Image 2.")
    return points1, points2

def compute_fundamental_matrix(points1, points2):
    """
    Compute the Fundamental Matrix using manually selected points.
    """
    print("\nStep 2: Compute the Fundamental Matrix.")
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    if F is None:
        raise ValueError("Unable to compute a robust Fundamental Matrix.")

    print("\nComputed Fundamental Matrix (F):\n", F)
    return F

def plot_epipolar_lines(img1, img2, F):
    """
    Plot the epipolar line in one image for a clicked point in the other image.
    """
    def on_click_image1(event):
        """
        Handle click events on Image 1 and draw the epipolar line in Image 2.
        """
        if event.xdata is None or event.ydata is None:
            return

        point1 = np.array([event.xdata, event.ydata, 1]).reshape(3, 1)
        line2 = F @ point1  # Epipolar line in Image 2

        ax2.clear()
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        draw_epipolar_line(ax2, line2, img2)
        ax2.set_title("Epipolar Line in Image 2")
        plt.draw()

    def on_click_image2(event):
        """
        Handle click events on Image 2 and draw the epipolar line in Image 1.
        """
        if event.xdata is None or event.ydata is None:
            return

        point2 = np.array([event.xdata, event.ydata, 1]).reshape(3, 1)
        line1 = F.T @ point2  # Epipolar line in Image 1

        ax1.clear()
        ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        draw_epipolar_line(ax1, line1, img1)
        ax1.set_title("Epipolar Line in Image 1")
        plt.draw()

    def draw_epipolar_line(ax, line, img):
        """
        Draw the epipolar line on the given image.
        """
        h, w = img.shape[:2]
        x = np.linspace(0, w, num=100)
        y = -(line[0] * x + line[2]) / line[1]
        ax.plot(x, y, color="red", label="Epipolar Line")
        ax.legend()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Click a point in Image 1")
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Click a point in Image 2")

    fig.canvas.mpl_connect("button_press_event", lambda event: on_click_image1(event) if event.inaxes == ax1 else on_click_image2(event))
    plt.show()

def compute_epipoles(F):
    """
    Compute epipoles from the Fundamental Matrix.
    """
    U, S, Vt = np.linalg.svd(F)
    e1 = Vt[-1] / Vt[-1, -1]
    e2 = U[:, -1] / U[-1, -1]
    print("\nEpipole in Image 1 (e1):", e1)
    print("Epipole in Image 2 (e2):", e2)
    return e1, e2

def plot_epipoles(img1, img2, e1, e2):
    """
    Plot the epipoles on the respective images.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.plot(e1[0], e1[1], 'ro', label="Epipole e1")
    ax1.legend()
    ax1.set_title("Epipole in Image 1")

    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.plot(e2[0], e2[1], 'ro', label="Epipole e2")
    ax2.legend()
    ax2.set_title("Epipole in Image 2")

    plt.show()

def main():
    img1 = cv2.imread(r"C:\Users\ftska\semester_5\Computer Vision\Lab assignments\ass_5\Images\Left camera image\WhatsApp Image 2024-11-21 at 17.37.27_5b15a237.jpg")
    img2 = cv2.imread(r"C:\Users\ftska\semester_5\Computer Vision\Lab assignments\ass_5\Images\Right camera image\WhatsApp Image 2024-11-21 at 17.37.40_a596f05b.jpg")

    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    points1, points2 = manually_select_points(img1, img2)
    F = compute_fundamental_matrix(points1, points2)
    plot_epipolar_lines(img1, img2, F)
    e1, e2 = compute_epipoles(F)
    plot_epipoles(img1, img2, e1, e2)

if __name__ == "__main__":
    main()
