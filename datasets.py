from pathlib import Path

from torch.utils.data import ConcatDataset
from torchvision.datasets import EMNIST as TorchEMNIST
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

__root_folder = Path(__file__).parent.resolve().as_posix()

EMNIST = ConcatDataset(
    [
        TorchEMNIST(
            root=f"{__root_folder}/datasets/",
            split="balanced",
            train=True,
            transform=ToTensor(),
        ),
        TorchEMNIST(
            root=f"{__root_folder}/datasets/",
            split="balanced",
            train=False,
            transform=ToTensor(),
        ),
    ]
)

DIDA = ImageFolder(root=f"{__root_folder}/datasets/DIDA", transform=ToTensor())

CARDIS = ImageFolder(root=f"{__root_folder}/datasets/CARDIS", transform=ToTensor())
