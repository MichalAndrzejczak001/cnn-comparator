import pytest
import torch

from backend.models.factory import create_model
from backend.models.simple_cnn import SimpleCNN
from backend.models.lenet5 import LeNet5
from backend.models.vgg11 import VGG11
from backend.models.resnet18_custom import ResNet18


class TestCreateModel:

    def test_returns_simple_cnn_instance(self):
        model = create_model("simple_cnn", num_classes=10, in_channels=1, input_size=(32, 32))
        assert isinstance(model, SimpleCNN)

    def test_returns_lenet5_instance(self):
        model = create_model("lenet5", num_classes=10, in_channels=1, input_size=(32, 32))
        assert isinstance(model, LeNet5)

    def test_returns_vgg11_instance(self):
        model = create_model("vgg11", num_classes=10, in_channels=3, input_size=(32, 32))
        assert isinstance(model, VGG11)

    def test_returns_resnet18_instance(self):
        model = create_model("resnet18", num_classes=10, in_channels=3, input_size=(32, 32))
        assert isinstance(model, ResNet18)

    def test_raises_value_error_for_unknown_name(self):
        with pytest.raises(ValueError):
            create_model("unknown_model", num_classes=10, in_channels=3, input_size=(32, 32))


class TestSimpleCNN:

    def test_output_shape_mnist(self):
        model = SimpleCNN(in_channels=1, num_classes=10, input_size=(32, 32))
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_cifar10(self):
        model = SimpleCNN(in_channels=3, num_classes=10, input_size=(32, 32))
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_num_classes(self):
        model = SimpleCNN(in_channels=1, num_classes=5, input_size=(32, 32))
        model.eval()
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_output_is_not_constant_for_different_inputs(self):
        model = SimpleCNN(in_channels=1, num_classes=10, input_size=(32, 32))
        model.eval()
        x1 = torch.randn(1, 1, 32, 32)
        x2 = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.equal(out1, out2)


class TestLeNet5:

    def test_output_shape_mnist(self):
        model = LeNet5(in_channels=1, num_classes=10, input_size=(32, 32))
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_cifar10(self):
        model = LeNet5(in_channels=3, num_classes=10, input_size=(32, 32))
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_num_classes(self):
        model = LeNet5(in_channels=1, num_classes=3, input_size=(32, 32))
        model.eval()
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_output_is_not_constant_for_different_inputs(self):
        model = LeNet5(in_channels=1, num_classes=10, input_size=(32, 32))
        model.eval()
        x1 = torch.randn(1, 1, 32, 32)
        x2 = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert not torch.equal(out1, out2)


class TestVGG11:

    def test_output_shape_cifar10(self):
        model = VGG11(in_channels=3, num_classes=10)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_mnist(self):
        model = VGG11(in_channels=1, num_classes=10)
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_num_classes(self):
        model = VGG11(in_channels=3, num_classes=20)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 20)


class TestResNet18:

    def test_output_shape_cifar10(self):
        model = ResNet18(in_channels=3, num_classes=10)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_mnist(self):
        model = ResNet18(in_channels=1, num_classes=10)
        model.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_custom_num_classes(self):
        model = ResNet18(in_channels=3, num_classes=5)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)
