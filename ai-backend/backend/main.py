import base64
import io
import os
import uuid

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm as mpl_cm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from torchvision import transforms

from backend.datasets.loader import load_dataset
from backend.models.factory import create_model
from backend.schemas import ExperimentConfig, CompareConfig, PredictResponse, ClassConfidence, GradCamResponse
from backend.training.trainer import train, evaluate

SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FASHION_MNIST_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
DATASET_CLASS_LABELS = {
    "mnist": MNIST_CLASSES,
    "cifar10": CIFAR10_CLASSES,
    "fashion_mnist": FASHION_MNIST_CLASSES,
}

app = FastAPI(
    title="CNN Comparator API",
    description="AI backend for comparing convolutional neural network architectures in image classification tasks",
    version="0.1.0",
)


def _resolve_dataset(dataset: str):
    if dataset == "mnist":
        return 1, (32, 32), 10, MNIST_CLASSES, transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    elif dataset == "fashion_mnist":
        return 1, (32, 32), 10, FASHION_MNIST_CLASSES, transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    elif dataset == "cifar10":
        return 3, (32, 32), 10, CIFAR10_CLASSES, transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    else:
        raise HTTPException(status_code=400, detail="Unknown dataset")


def _get_target_layer(model, model_name: str):
    if model_name == "simple_cnn":
        return model.conv2
    elif model_name == "lenet5":
        return model.conv2
    elif model_name == "vgg11":
        return model.features[18]
    elif model_name == "resnet18":
        return model.model.layer4[-1]
    elif model_name == "alexnet":
        return model.features[10]
    elif model_name == "mobilenet":
        return model.dw6
    raise ValueError(f"Unknown model: {model_name}")


def _compute_grad_cam(model, tensor, target_layer, predicted_index, device):
    activations, gradients = [], []

    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    output = model(tensor.to(device))
    model.zero_grad()
    output[0, predicted_index].backward()

    fh.remove()
    bh.remove()

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1)).squeeze().detach().cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def _overlay_cam(rgb_img, cam):
    H, W = rgb_img.shape[:2]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ) / 255.0
    heatmap = mpl_cm.jet(cam_resized)[:, :, :3].astype(np.float32)
    blended = (0.5 * rgb_img + 0.5 * heatmap).clip(0, 1)
    return (blended * 255).astype(np.uint8)


def _generate_sample_gradcams(model, model_name, test_loader, class_labels, device):
    model.eval()
    samples = {}
    n_classes = len(class_labels)

    for images, labels in test_loader:
        for img, label in zip(images, labels):
            idx = label.item()
            if idx not in samples:
                samples[idx] = img.unsqueeze(0)
            if len(samples) == n_classes:
                break
        if len(samples) == n_classes:
            break

    target_layer = _get_target_layer(model, model_name)
    result = []

    for true_idx, tensor in sorted(samples.items()):
        try:
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = F.softmax(logits, dim=1).squeeze().tolist()
            pred_idx = int(torch.argmax(torch.tensor(probs)).item())
            confidence = round(probs[pred_idx], 4)

            cam = _compute_grad_cam(model, tensor, target_layer, pred_idx, device)

            img_np = tensor.squeeze().cpu().numpy()
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            else:
                img_np = np.stack([img_np] * 3, axis=-1)

            overlay = _overlay_cam(img_np.astype(np.float32), cam)

            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")

            result.append({
                "true_label": class_labels[true_idx],
                "predicted_label": class_labels[pred_idx],
                "confidence": confidence,
                "gradcam_image": base64.b64encode(buf.getvalue()).decode(),
            })
        except Exception:
            pass

    return result


@app.post("/experiments")
def run_experiment(config: ExperimentConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes, in_channels, input_size = load_dataset(
        config.dataset, config.training.batch_size
    )
    model = create_model(config.model, num_classes, in_channels, input_size)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_loss, test_loss_per_epoch, training_time = train(
        model, train_loader, test_loader, config.training.epochs, optimizer, device=device
    )
    metrics = evaluate(model, test_loader, device=device)

    model_id = str(uuid.uuid4())
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth"))

    sample_gradcams = _generate_sample_gradcams(
        model, config.model, test_loader, DATASET_CLASS_LABELS.get(config.dataset, []), device
    )

    return {
        "status": "training and evaluation finished",
        "model_id": model_id,
        "train_loss_per_epoch": train_loss,
        "test_loss_per_epoch": test_loss_per_epoch,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "training_time_seconds": training_time,
        "sample_gradcams": sample_gradcams,
    }


@app.post("/compare")
def compare_models(config: CompareConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, num_classes, in_channels, input_size = load_dataset(
        config.dataset, config.training.batch_size
    )

    results = []
    for model_name in ["simple_cnn", "lenet5", "alexnet", "vgg11", "resnet18", "mobilenet"]:
        model = create_model(model_name, num_classes, in_channels, input_size)
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

        train_loss, test_loss_per_epoch, training_time = train(
            model, train_loader, test_loader, config.training.epochs, optimizer, device=device
        )
        metrics = evaluate(model, test_loader, device=device)

        results.append({
            "model": model_name,
            "train_loss_per_epoch": train_loss,
            "test_loss_per_epoch": test_loss_per_epoch,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "training_time_seconds": training_time,
            "confusion_matrix": metrics["confusion_matrix"],
        })

    return {
        "dataset": config.dataset,
        "epochs": config.training.epochs,
        "results": results,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    model_name: str = Form(...),
    dataset: str = Form(...),
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    weights_path = os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Model weights not found")

    in_channels, input_size, num_classes, class_labels, transform = _resolve_dataset(dataset)

    image = Image.open(io.BytesIO(file.file.read())).convert("RGB" if in_channels == 3 else "L")
    tensor = transform(image).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name, num_classes, in_channels, input_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze().tolist()

    predicted_index = int(torch.argmax(torch.tensor(probs)).item())
    confidences = [
        ClassConfidence(label=class_labels[i], confidence=round(probs[i], 6))
        for i in range(num_classes)
    ]

    return PredictResponse(
        predicted_class=class_labels[predicted_index],
        predicted_index=predicted_index,
        confidences=confidences,
    )


@app.post("/gradcam", response_model=GradCamResponse)
def gradcam(
    model_name: str = Form(...),
    dataset: str = Form(...),
    model_id: str = Form(...),
    file: UploadFile = File(...),
):
    weights_path = os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Model weights not found")

    in_channels, input_size, num_classes, class_labels, transform = _resolve_dataset(dataset)

    image_bytes = file.file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))
    tensor = transform(image_pil.convert("L" if in_channels == 1 else "RGB")).unsqueeze(0)
    rgb_img = np.array(image_pil.convert("RGB").resize((32, 32))).astype(np.float32) / 255.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name, num_classes, in_channels, input_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        probs = F.softmax(model(tensor.to(device)), dim=1).squeeze().tolist()

    predicted_index = int(torch.argmax(torch.tensor(probs)).item())
    confidences = [
        ClassConfidence(label=class_labels[i], confidence=round(probs[i], 6))
        for i in range(num_classes)
    ]

    cam = _compute_grad_cam(model, tensor, _get_target_layer(model, model_name), predicted_index, device)
    visualization = _overlay_cam(rgb_img, cam)

    buf = io.BytesIO()
    Image.fromarray(visualization).save(buf, format="PNG")

    return GradCamResponse(
        predicted_class=class_labels[predicted_index],
        predicted_index=predicted_index,
        confidences=confidences,
        gradcam_image=base64.b64encode(buf.getvalue()).decode(),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
