import time
import torch
import torch.nn as nn


def train(model, loader, epochs, optimizer, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

    training_time = round(time.time() - start, 2)
    return history, training_time


def evaluate(model, loader, device="cpu"):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    num_classes = max(max(all_labels), max(all_preds)) + 1
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": matrix
    }


