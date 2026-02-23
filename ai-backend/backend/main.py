from fastapi import FastAPI
from backend.schemas import ExperimentConfig, CompareConfig
import torch
from backend.training.trainer import train, evaluate
from backend.models.factory import create_model
from backend.datasets.loader import load_dataset
import torch.optim as optim

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

    train_loss = train(
        model,
        train_loader,
        config.training.epochs,
        optimizer,
        device=device
    )

    metrics = evaluate(
        model,
        test_loader,
        device=device
    )

    return {
        "status": "training and evaluation finished",
        "train_loss_per_epoch": train_loss,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"]
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

        train_loss = train(
            model,
            train_loader,
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
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"]
        })

    return {
        "dataset": config.dataset,
        "epochs": config.training.epochs,
        "results": results
    }



@app.get("/health")
def health():
    return {"status": "ok"}
