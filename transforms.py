import cv2
import numpy as np


def emnist_transform(image: np.ndarray) -> np.ndarray:
    # Greyscale
    t = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    t: cv2.Mat = cv2.GaussianBlur(t, sigmaX=1)
    # Extract the ROI
    # ...
    # Pad the ROI
    t = cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    # Center the image
    roi_h, roi_w = t.shape[:2]
    square_size = max(roi_h, roi_w)
    # Create a blank square canvas (black background)
    canvas = np.zeros((square_size, square_size), dtype=np.uint8)
    # Calculate the offsets to center the ROI in the square
    x_offset = (square_size - roi_w) // 2
    y_offset = (square_size - roi_h) // 2
    # Place the ROI onto the center of the canvas
    canvas[y_offset : y_offset + roi_h, x_offset : x_offset + roi_w] = t
    t = canvas
    # Downsample to 28 x 28
    t = cv2.resize(t, (28, 28), interpolation=cv2.INTER_CUBIC)

    return t
