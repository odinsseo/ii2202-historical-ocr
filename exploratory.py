from typing import Tuple

import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_pixel_frequency(
    dataset: ImageFolder, num_pixels=256, normalize=True
) -> np.ndarray:
    """
    Computes the pixel frequency distribution for a PyTorch image dataset.

    Parameters:
        dataset (ImageFolder): A PyTorch `ImageFolder` dataset containing images, assumed to be grayscale or single-channel.
        num_pixels (int, optional): The number of bins for the histogram, typically 256 for 8-bit images. Default is 256.
        normalize (bool, optional): Whether to normalize the pixel frequencies such that they sum to 1. Default is True.

    Returns:
        np.ndarray: A NumPy array of shape `(num_pixels,)` containing the pixel frequency distribution.
                    If `normalize` is True, the values will sum to 1.

    Notes:
        The function is designed for single-channel (grayscale) images.
    """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize histogram array to accumulate pixel counts across the dataset
    histogram = np.zeros(num_pixels, dtype=int)

    for images, _ in dataloader:
        images: Tensor

        # Flatten images and count pixel values
        images_flattened = images.view(-1).numpy()
        pixel_values = (images_flattened * (num_pixels - 1)).astype(int)
        histogram += np.bincount(pixel_values, minlength=num_pixels)

    # Normalize histogram if needed
    if normalize:
        histogram = histogram / histogram.sum()

    return histogram


def get_class_avarage_image(
    dataset: ImageFolder, num_classes: int, image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Calculates and plots the average pixel values for each class in a PyTorch image dataset.

    Parameters:
        dataset (ImageFolder): A PyTorch `ImageFolder` dataset containing images and their labels.
        num_classes (int): The total number of classes in the dataset (e.g., the number of digits and characters).
        image_shape (Tuple[int, int]): The shape of each image as a tuple of (height, width).

    Returns:
        np.ndarray: A NumPy array of shape `(num_classes, height, width)`,
                    where each entry contains the average pixel values for the corresponding class.

    Notes:
        Assumes each image has been preprocessed to have a single channel (e.g., grayscale).
    """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize arrays to accumulate pixel values and counts for each class
    pixel_sums = np.zeros((num_classes, *image_shape), dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=int)

    for images, labels in dataloader:
        images: Tensor

        images = images.numpy()  # Convert to NumPy array
        for img, label in zip(images, labels):
            pixel_sums[label] += img.squeeze()  # Accumulate pixel values
            class_counts[label] += 1  # Count samples per class

    # Calculate average pixel value for each class
    average_pixels = pixel_sums / class_counts[:, None, None]

    return average_pixels


def get_distances_from_mean(
    dataset: ImageFolder, avarage_pixels: np.ndarray
) -> np.ndarray:
    """
    Calculates the Euclidean distances of each image in a dataset from the average pixel values (mean image)
    for its corresponding class.

    Args:
        dataset (ImageFolder): A PyTorch `ImageFolder` dataset containing the images organized in class-specific folders.
                               Each image's class label is determined by the directory it belongs to.
        avarage_pixels (np.ndarray): A NumPy array where each entry represents the mean pixel values (mean image)
                                     of a class in the dataset. The array shape should match the number of classes.

    Returns:
        np.ndarray: A dictionary mapping each class label (as a string) to a list of Euclidean distances.
                    Each list contains distances of images in that class from the mean image of the class.
    """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    mean_images = avarage_pixels

    # Step 2: Calculate Euclidean distance for each image to its class mean
    distances = {cls: [] for cls in dataset.class_to_idx.keys()}
    for images, labels in dataloader:
        images: Tensor
        labels: Tensor

        images = images.numpy()  # Convert to NumPy array
        labels = labels.numpy()  # Convert labels to NumPy array
        for img, label in zip(images, labels):
            # Calculate Euclidean distance to the mean image of the class
            diff = img.squeeze() - mean_images[label]
            distance = np.sqrt(np.sum(diff**2))
            distances[idx_to_class[label]].append(distance)

    return distances


def get_class_intensity(
    dataset: ImageFolder,
) -> DataFrame:
    """
    Calculates the average pixel intensity for each image in a PyTorch dataset and returns a DataFrame
    containing class names and their respective average intensities.

    Parameters:
        dataset (ImageFolder): A PyTorch `ImageFolder` dataset containing grayscale or single-channel images with labels.

    Returns:
        DataFrame: A pandas DataFrame with two columns
        - "intensity": Contains the average pixel intensity (float) for each image in the dataset.
        - "class": Contains the class name (str) corresponding to each image.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Compute the average pixel intensity for each image
    intensities = []
    classes = []

    for images, target in loader:
        images: Tensor
        target: Tensor

        for i in range(images.size(0)):
            img = images[i]  # Get the individual image tensor
            label = target[i].item()  # Get the label for the image

            # Compute the average pixel intensity
            avg_intensity = (
                img.mean().item()
            )  # Mean pixel intensity (over all channels if RGB, or grayscale)

            intensities.append(avg_intensity)
            classes.append(idx_to_class[label])

    return DataFrame({"intensity": intensities, "class": classes})
