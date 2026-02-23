import torch
import torch.nn as nn


def train(model, loader, epochs, optimizer, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    history = []

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

    return history


def evaluate(model, loader, device="cpu"):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }


