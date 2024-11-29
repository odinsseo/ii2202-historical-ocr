from pathlib import Path
from typing import Literal

import numpy as np
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import EMNIST as TorchEMNIST
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    Grayscale,
    InterpolationMode,
    Lambda,
    Normalize,
    Resize,
    ToTensor,
)

from transforms import emnist_transform

__root_folder = Path(__file__).parent.resolve().as_posix()

EMNIST_TRAIN = TorchEMNIST(
    root=f"{__root_folder}/datasets/",
    split="balanced",
    train=True,
    transform=ToTensor(),
)
EMNIST_TEST = TorchEMNIST(
    root=f"{__root_folder}/datasets/",
    split="balanced",
    train=False,
    transform=Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
        ]
    ),
)


def invert_image(x: Tensor) -> Tensor:
    return 1.0 - x


def basic_transform(x: np.ndarray) -> np.ndarray:
    return emnist_transform(x, roi=False, invert=True)


def roi_transform(x: np.ndarray) -> np.ndarray:
    return emnist_transform(x, roi=True, invert=True)


def binary_transform(x: np.ndarray) -> np.ndarray:
    return emnist_transform(x, roi=True, invert=True, binary=True)


def to_numpy(x: Image) -> np.ndarray:
    return np.array(x)


def get_historical_dataset(
    type: Literal["raw", "basic", "roi", "binary"]
) -> ImageFolder:
    match type:
        case "raw":
            transform = Compose(
                [
                    Grayscale(1),
                    Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
                    ToTensor(),
                    Lambda(invert_image),
                    Normalize((0.1307,), (0.3081,)),
                ]
            )
        case "basic":
            transform = Compose([Lambda(to_numpy), Lambda(basic_transform), ToTensor()])
        case "roi":
            transform = Compose(
                [
                    Lambda(to_numpy),
                    Lambda(roi_transform),
                    ToTensor(),
                    Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
                    Normalize((0.1307,), (0.3081,)),
                ]
            )
        case "binary":  # non-inverted
            transform = Compose(
                [
                    Lambda(to_numpy),
                    Lambda(binary_transform),
                    ToTensor(),
                    Resize((28, 28), interpolation=InterpolationMode.BICUBIC),
                    Normalize((0.1307,), (0.3081,)),
                ]
            )
        case _:
            raise ValueError(f'"{type}" not supported.')

    historical = ImageFolder(
        f"{__root_folder}/datasets/HISTORICAL", transform=transform
    )
    historical.class_to_idx = EMNIST_TRAIN.class_to_idx

    return historical
