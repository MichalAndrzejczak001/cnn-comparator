from .simple_cnn import SimpleCNN
from .lenet5 import LeNet5


def create_model(
    name: str,
    num_classes: int,
    in_channels: int,
    input_size: tuple
):
    if name == "simple_cnn":
        if in_channels != 1:
            raise ValueError("SimpleCNN supports only 1-channel input")
        return SimpleCNN(num_classes)

    elif name == "lenet5":
        return LeNet5(
            in_channels=in_channels,
            num_classes=num_classes,
            input_size=input_size
        )

    else:
        raise ValueError("Unknown model")

