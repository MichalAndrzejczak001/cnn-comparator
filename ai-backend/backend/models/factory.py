from .simple_cnn import SimpleCNN

def create_model(name: str, num_classes: int):
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError("Unknown model")
