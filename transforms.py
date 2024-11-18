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

    return blur


def roi_process(img: np.ndarray) -> np.ndarray:

    return img



def emnist_transform(image: np.ndarray) -> np.ndarray:

    gray = greyscale_and_denoising(image)

    # Extract the ROI
    edges = cv2.Canny(gray.astype(np.uint8), 30, 150)

    # Display the results
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, cnts, -1, (0, 255, 0), 1)

    largest_contour = max(cnts, key=cv2.contourArea)

    chars = []

    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)


    side_length = max(w, h)  # Use the larger dimension for square
    center_x = x + w // 2
    center_y = y + h // 2
    x_start = max(center_x - side_length // 2, 0)
    y_start = max(center_y - side_length // 2, 0)
    x_end = min(center_x + side_length // 2, gray.shape[1])
    y_end = min(center_y + side_length // 2, gray.shape[0])

    # // filter out bounding boxes, ensuring they are neither too small
    # // nor too large
    # //if (w >= 5 and w <= 150) and (h >= 15 and h <= 120): --> take out for now as it is not working
    # extract the character and threshold it to make the character
    # appear as *white* (foreground) on a *black* background, then
    # grab the width and height of the thresholded image
    roi = gray[y_start:y_end, x_start:x_end]

    square_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_CUBIC)
    normalized = square_roi.astype("float32") / 255.0



    # //(tH, tW) = roi.shape
    # if the width is greater than the height, resize along the
    # width dimension

    # //if tW > tH:
    # //    resized_roi = imutils.resize(roi, width=32, inter=cv2.INTER_CUBIC)
    # //else:
    # //    resized_roi = imutils.resize(roi, height=32, inter=cv2.INTER_CUBIC)


    # re-grab the image dimensions (now that its been resized)
    # and then determine how much we need to pad the width and
    # height such that our image will be 28x28
    # //(tH, tW) = square_roi.shape

    # //dX = int(max(0, 28 - tW) / 2.0)
    # //dY = int(max(0, 28 - tH) / 2.0)

    # //# Calculate the background intensity (e.g., average intensity)
    # //background_intensity = int(np.median(gray)) 
# //
    # //# pad the image and force 28x28 dimensions
    # //padded = cv2.copyMakeBorder(
    # //    resized_roi,
    # //    top=dY,
    # //    bottom=dY,
    # //    left=dX,
    # //    right=dX,
    # //    borderType=cv2.BORDER_CONSTANT,
    # //    value=background_intensity,
    # //)
    # //padded = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_CUBIC)
    # //# prepare the padded image for classification via our
    # //# handwriting OCR model
    # //padded = padded.astype("float32") / 255.0
    # //# update our list of characters that will be OCR'd
    # //chars.append((padded, (x, y, w, h)))
# //
    # //transfor_img = chars[0][0].squeeze()
    plot_steps([image, gray, edges, contour_image, roi, normalized])

    return chars


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