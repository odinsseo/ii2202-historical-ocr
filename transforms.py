import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours


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


def emnist_transform(image: np.ndarray) -> np.ndarray:
    # Greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize (not in EMNIST)
    # gray = cv2.equalizeHist((gray * 255).astype(np.uint8))

    # Gaussian blur
    blurred: cv2.Mat = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)

    # Extract the ROI
    edged = cv2.Canny((blurred * 255).astype(np.uint8), 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y : y + h, x : x + w]
            thresh = cv2.threshold(
                (roi * 255).astype(np.uint8),
                0,
                255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
            )[1]
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            thresh = (thresh * 255).astype(np.uint8)
            if tW > tH:
                thresh = imutils.resize(thresh, width=32, inter=cv2.INTER_CUBIC)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32, inter=cv2.INTER_CUBIC)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 28x28
            (tH, tW) = thresh.shape
            dX = int(max(0, 28 - tW) / 2.0)
            dY = int(max(0, 28 - tH) / 2.0)
            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(
                thresh,
                top=dY,
                bottom=dY,
                left=dX,
                right=dX,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            padded = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_CUBIC)
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    return chars
