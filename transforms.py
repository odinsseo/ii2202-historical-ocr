import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def increase_contrast(image: np.ndarray) -> np.ndarray:
    # converting to LAB color space
    lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def plot_steps(imgs):

    _, axs = plt.subplots(1, len(imgs), figsize=(15, 5))

    for i,img in enumerate(imgs): 
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


# ~ STEP ONE (Greyscale)
def greyscale_and_denoising(img: np.ndarray, sigmaX: float = 1.0) -> np.ndarray:
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
    stretched = ((blur - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return stretched


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
    cv2.drawContours(contour_image, cnts, -1, (0, 255, 0), 1)

    # Find the largest contour by area
    largest_contour = max(cnts, key=lambda c: cv2.arcLength(c, closed=False))

    return largest_contour, edges, contour_image


# ~ STEP TRHEE (OPT1: Dynamic ROI)
def dynamic_roi(img: np.ndarray, coor: tuple, size: int = 28) -> np.ndarray:
    """
    Extracts and dynamically resizes a Region of Interest (ROI) from an image 
    based on the given bounding box coordinates, ensuring the final output is 28x28 pixels.

    Args:
        img (np.ndarray): Input image (grayscale or single-channel image expected).
        coor (tuple): A tuple of (x, y, w, h) defining the bounding box of the ROI.
    Returns:
        np.ndarray: The processed ROI as a 28x28 grayscale image.
    """

    (x, y, w, h) = coor

    #Extract the ROI dynamically based on the bounding box
    roi = img[y : y + h, x : x + w]

    #Resize dynamically based on the largest contour
    tH, tW = roi.shape

    if tW > tH:
        resized_roi = cv2.resize(roi, (32, int(32 * tH / tW)), interpolation=cv2.INTER_CUBIC)
    else:
        resized_roi = cv2.resize(roi, (int(32 * tW / tH), 32), interpolation=cv2.INTER_CUBIC)

    #Recalculate dimensions after resizing
    tH, tW = resized_roi.shape

    #Padding to enforce 28x28 size
    dX = max((28 - tW) // 2, 0)
    dY = max((28 - tH) // 2, 0)
    
    #Calculate the background intensity (e.g., median intensity)
    background_intensity = int(np.median(img))

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized_roi,
        top=dY,
        bottom=dY,
        left=dX,
        right=dX,
        borderType=cv2.BORDER_CONSTANT,
        value=background_intensity,
    )

    #Final resize to enforce exact 28x28 dimensions (if needed)
    padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)

    return padded


# ~ STEP THREE (OPT2: Box ROI)
def box_roi_and_resizing(img: np.ndarray, coor: tuple, size: int = 28) -> np.ndarray:
    """
    Crop an ROI to a square based on bounding box and resize to 28x28.

    Args:
        img (np.ndarray): Input image.
        coor (tuple): Bounding box coordinates as (x, y, w, h).

    Returns:
        np.ndarray: Resized square ROI (28x28).
    """

    (x, y, w, h) = coor

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
    square_roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)

    return square_roi



def emnist_transform(image: np.ndarray) -> np.ndarray:

    grey = greyscale_and_denoising(image)

    # change levels to black 
    

    largest_contour, edges, contour_image = edge_extraction(grey)

    # compute the bounding box of the contour
    coor = cv2.boundingRect(largest_contour)

    # call roi function to adapt to letter size and crop the image to 28x28 pixels
    roi_img_box = box_roi_and_resizing(grey, coor)
    roi_img_dynamic = dynamic_roi(grey, coor)

    # //normalized_box = roi_img_box.astype("float32") / 255.0
    # //normalized_dynamic = roi_img_dynamic.astype("float32") / 255.0

    plot_steps([image, grey, edges, contour_image, roi_img_box, roi_img_dynamic])

    return grey


def emnist_transform2(image: np.ndarray) -> np.ndarray:
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image * 255).astype('uint8')

    # Step 1: Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 2: Threshold the image
    _, binary_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

    # Step 3: Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    cleaned_image = cleaned_image.astype('uint8')

    # Step 4: Filter small contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned_image)

    print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1250:  # Keep only large contours
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Final cleaned image
    final_image = cv2.bitwise_and(cleaned_image, mask)

    # Display the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title('Binary Image')
    axs[1].axis('off')

    axs[2].imshow(final_image, cmap='gray')
    axs[2].set_title('Cleaned Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    return final_image