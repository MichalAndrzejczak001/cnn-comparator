from fastapi import FastAPI
from backend.models.factory import create_model
from backend.datasets.loader import load_dataset
from backend.training.trainer import train
from backend.schemas import ExperimentConfig
import torch.optim as optim

app = FastAPI(
    title="CNN Comparator API",
    description="AI backend for a web application for comparing convolutional neural network architectures in image classification tasks",
    version="0.1.0"
)

@app.post("/experiments")
def run_experiment(config: ExperimentConfig):
    train_loader, _, num_classes, in_channels = load_dataset(
        config.dataset,
        config.training.batch_size
    )

    input_size = (28, 28) if config.dataset == "mnist" else (32, 32)

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

    history = train(
        model,
        train_loader,
        config.training.epochs,
        optimizer
    )

    return {
        "status": "training finished",
        "loss_per_epoch": history
    }

@app.get("/health")
def health():
    return {"status": "ok"}
