from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from backend.schemas import ExperimentConfig, CompareConfig, PredictResponse, ClassConfidence
import torch
import torch.nn.functional as F
from backend.training.trainer import train, evaluate
from backend.models.factory import create_model
from backend.datasets.loader import load_dataset
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import io
import uuid
import os

SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

app = FastAPI(
    title="CNN Comparator API",
    description="AI backend for a web application for comparing convolutional neural network architectures in image classification tasks",
    version="0.1.0"
)


@app.post("/experiments")
def run_experiment(config: ExperimentConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes, in_channels, input_size = load_dataset(
        config.dataset,
        config.training.batch_size
    )

    model = create_model(
        config.model,
        num_classes,
        in_channels,
        input_size
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate
    )

    train_loss, test_loss_per_epoch, training_time = train(
        model,
        train_loader,
        test_loader,
        config.training.epochs,
        optimizer,
        device=device
    )

    metrics = evaluate(
        model,
        test_loader,
        device=device
    )

    model_id = str(uuid.uuid4())
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth"))

    return {
        "status": "training and evaluation finished",
        "model_id": model_id,
        "train_loss_per_epoch": train_loss,
        "test_loss_per_epoch": test_loss_per_epoch,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "training_time_seconds": training_time
    }

@app.post("/compare")
def compare_models(config: CompareConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes, in_channels, input_size = load_dataset(
        config.dataset,
        config.training.batch_size
    )

    results = []

    for model_name in ["simple_cnn", "lenet5", "vgg11", "resnet18"]:
        model = create_model(
            model_name,
            num_classes,
            in_channels,
            input_size
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate
        )

        train_loss, test_loss_per_epoch, training_time = train(
            model,
            train_loader,
            test_loader,
            config.training.epochs,
            optimizer,
            device=device
        )

        metrics = evaluate(
            model,
            test_loader,
            device=device
        )

        results.append({
            "model": model_name,
            "train_loss_per_epoch": train_loss,
            "test_loss_per_epoch": test_loss_per_epoch,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "training_time_seconds": training_time,
            "confusion_matrix": metrics["confusion_matrix"]
        })

    return {
        "dataset": config.dataset,
        "epochs": config.training.epochs,
        "results": results
    }



@app.post("/predict", response_model=PredictResponse)
def predict(
    model_name: str = Form(...),
    dataset: str = Form(...),
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    weights_path = os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Model weights not found")

    if dataset == "mnist":
        in_channels, input_size, num_classes = 1, (32, 32), 10
        class_labels = MNIST_CLASSES
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    elif dataset == "cifar10":
        in_channels, input_size, num_classes = 3, (32, 32), 10
        class_labels = CIFAR10_CLASSES
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    else:
        raise HTTPException(status_code=400, detail="Unknown dataset")

    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB" if in_channels == 3 else "L")
    tensor = transform(image).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name, num_classes, in_channels, input_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    predicted_index = int(torch.argmax(torch.tensor(probs)).item())
    confidences = [
        ClassConfidence(label=class_labels[i], confidence=round(probs[i], 6))
        for i in range(num_classes)
    ]

    return PredictResponse(
        predicted_class=class_labels[predicted_index],
        predicted_index=predicted_index,
        confidences=confidences
    )


@app.get("/health")
def health():
    return {"status": "ok"}
