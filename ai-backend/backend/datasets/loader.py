from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def load_dataset(name: str, batch_size: int):
    if name != "mnist":
        raise ValueError("Unknown dataset")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = MNIST("./data", train=True, download=True, transform=transform)
    test = MNIST("./data", train=False, download=True, transform=transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
        10
    )
