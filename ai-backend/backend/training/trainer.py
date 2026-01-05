import torch
import torch.nn as nn

def train(model, loader, epochs, optimizer, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

    return history

