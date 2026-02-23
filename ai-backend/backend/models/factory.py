from .simple_cnn import SimpleCNN
from .lenet5 import LeNet5
from .vgg11 import VGG11

from .resnet18_custom import ResNet18

def create_model(name, num_classes, in_channels, input_size):
    if name == "simple_cnn":
        return SimpleCNN(in_channels, num_classes, input_size)

    elif name == "lenet5":
        return LeNet5(in_channels, num_classes, input_size)

    elif name == "vgg11":
        return VGG11(in_channels, num_classes)

    elif name == "resnet18":
        return ResNet18(in_channels, num_classes)

    else:
        raise ValueError("Unknown model")



