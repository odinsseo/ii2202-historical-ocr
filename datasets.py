from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import EMNIST as TorchEMNIST
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
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


class HistoricalImageFolder(ImageFolder):
    img_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        allow_empty=False,
    ):
        super().__init__(
            root, transform, target_transform, loader, is_valid_file, allow_empty
        )
        self.classes = [cls.upper() for cls in self.classes]
        self.class_to_idx = {
            cls: EMNIST_TRAIN.class_to_idx[cls] for cls in self.classes
        }

        self.samples = self.make_dataset(
            self.root,
            class_to_idx=self.class_to_idx,
            extensions=HistoricalImageFolder.img_extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, str]:
        sample, target = super().__getitem__(index)

        return sample, target, self.samples[index][0]


def get_historical_dataset(
    type: Literal["raw", "basic", "roi", "binary"]
) -> HistoricalImageFolder:
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

    historical = HistoricalImageFolder(
        f"{__root_folder}/datasets/DIDA-clean", transform=transform
    )

    return historical
