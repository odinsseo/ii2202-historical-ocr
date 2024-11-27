import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_steps(imgs, titles):

    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 5))

    for i, img in enumerate(imgs):
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(titles[i])


    fig.align_titles()
    plt.tight_layout()
    plt.show()


# ~ STEP ONE (Greyscale)
def greyscale_and_denoising(img: np.ndarray, binary: bool = False, sigmaX: float = 1.0) -> np.ndarray:
    """
    Convert an image to grayscale, normalize intensity values, and apply Gaussian blur for denoising.
    Args:
        img (np.ndarray): Input image (BGR).
        sigmaX (float): Standard deviation for Gaussian blur. Default is 1.0.
    Returns:
        np.ndarray: Preprocessed grayscale and denoised image.
    """

    # Greyscale to reduce from three chanel to one channel
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalization: lower intesity value is assigned 0 and higest 255
    normalize = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX)

    # Denoising
    blur = cv2.GaussianBlur(normalize, (0, 0), sigmaX=sigmaX)

    # Stretch contrast
    min_val, max_val = np.min(blur), np.max(blur)
    final_img = ((blur - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    if binary:
        # Switch to binary
        _, final_img = cv2.threshold(final_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return final_img


# ~ STEP TWO (Edge extraction)
def edge_extraction(img: np.ndarray, c_th1: int = 30, c_th2: int = 150) -> tuple:
    """
    Extract the largest contour from an image and return the contour, edges,
    and an annotated image showing the contours.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        tuple: (largest_contour, edges, contour_image)
            - largest_contour: The largest contour detected.
            - edges: The edges detected using Canny edge detection.
            - contour_image: Image with contours drawn.
    """

    # Edge detection using Canny
    edges = cv2.Canny(img.astype(np.uint8), c_th1, c_th2)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image to visualize contours
    contour_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
    # Find the largest contour by area
    largest_contour = max(cnts, key=lambda c: cv2.arcLength(c, closed=False)) if cnts else []

    return largest_contour, edges, contour_image


# ~ STEP TRHEE (OPT1: Dynamic ROI)
def dynamic_roi(img: np.ndarray, largest_contour: any, size: int = 28) -> np.ndarray:
    """
    Extracts and dynamically resizes a Region of Interest (ROI) from an image
    based on the given bounding box coordinates, ensuring the final output is 28x28 pixels.

    Args:
        img (np.ndarray): Input image (grayscale or single-channel image expected).
        coor (tuple): A tuple of (x, y, w, h) defining the bounding box of the ROI.
    Returns:
        np.ndarray: The processed ROI as a 28x28 grayscale image.
    """

    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Extract the ROI dynamically based on the bounding box
    roi = img[y : y + h, x : x + w]

    # Resize dynamically based on the largest contour
    tH, tW = roi.shape

    if tW > tH:
        resized_roi = cv2.resize(
            roi, (32, int(32 * tH / tW)), interpolation=cv2.INTER_CUBIC
        )
    else:
        resized_roi = cv2.resize(
            roi, (int(32 * tW / tH), 32), interpolation=cv2.INTER_CUBIC
        )

    # Recalculate dimensions after resizing
    tH, tW = resized_roi.shape

    # Padding to enforce 28x28 size
    dX = max((28 - tW) // 2, 0)
    dY = max((28 - tH) // 2, 0)

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized_roi,
        top=dY,
        bottom=dY,
        left=dX,
        right=dX,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )

    # Final resize to enforce exact 28x28 dimensions (if needed)
    padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)

    return padded


# ~ STEP THREE (OPT2: Box ROI)
def box_roi_and_resizing(
    img: np.ndarray, largest_contour: any, size: int = 28
) -> np.ndarray:
    """
    Crop an ROI to a square based on bounding box and resize to 28x28.

    Args:
        img (np.ndarray): Input image.
        coor (tuple): Bounding box coordinates as (x, y, w, h).

    Returns:
        np.ndarray: Resized square ROI (28x28).
    """

    # TODO: invetigate this step more if tehre is time: Create a mask for the largest contour
    # //mask = np.zeros_like(img)
    # //cv2.drawContours(img, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the bounding box of the largest contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Crop the ROI around the bounding box
    roi = img[y : y + h, x : x + w]

    # Calculate square bounding box
    side_length = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    x_start = max(center_x - side_length // 2, 0)
    y_start = max(center_y - side_length // 2, 0)
    x_end = min(center_x + side_length // 2, img.shape[1])
    y_end = min(center_y + side_length // 2, img.shape[0])

    # Crop and resize
    roi = img[y_start:y_end, x_start:x_end]

    padded_img = cv2.copyMakeBorder(
        roi,
        top=2,
        bottom=2,
        left=2,
        right=2,
        borderType=cv2.BORDER_CONSTANT,
        value=int(np.max(img)),
    )

    square_roi = cv2.resize(padded_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return square_roi


def emnist_transform(
    image: np.ndarray, roi: bool = True, invert: bool = True, binary = False
) -> np.ndarray:

    grey = greyscale_and_denoising(image, binary)

    if roi: 
        largest_contour, _, _ = edge_extraction(grey)

    if largest_contour is not None and len(largest_contour) > 0:
        # call roi function to adapt to letter size and crop the image to 28x28 pixels
        roi_img = box_roi_and_resizing(grey, largest_contour)
        #roi_img = dynamic_roi(grey, largest_contour)
    else:
        roi_img = grey

    final_img = roi_img.astype(np.float32) / 255.0

    if invert:
        # Invert intensity
        final_img = 1.0 - final_img

    # //titles = ["(a) Original", "(b) Greyscale", "(f) ROI", "(g) Inverted"]
    # //images = [image, grey, roi_img, final_img]
    # //plot_steps(images, titles)

    return final_img
