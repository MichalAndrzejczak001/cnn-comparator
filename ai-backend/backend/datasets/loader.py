from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple


def load_dataset(name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Zwraca:
    - train_loader
    - test_loader
    - num_classes
    - in_channels
    """

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train = MNIST("./data", train=True, download=True, transform=transform)
        test = MNIST("./data", train=False, download=True, transform=transform)

        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size),
            10,   # num_classes
            1     # in_channels
        )

    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train = CIFAR10("./data", train=True, download=True, transform=transform)
        test = CIFAR10("./data", train=False, download=True, transform=transform)

        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size),
            10,   # num_classes
            3     # RGB
        )

    else:
        raise ValueError("Unknown dataset")
