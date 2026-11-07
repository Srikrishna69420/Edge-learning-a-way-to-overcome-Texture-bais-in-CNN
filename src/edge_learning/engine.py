"""Training and evaluation routines."""

import torch
from torch import nn


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": 100.0 * correct / max(total, 1)}


def train_one_epoch(model, loader, optimizer, device, edge_weight=0.7):
    model.train()
    classification_loss = nn.CrossEntropyLoss()
    edge_loss = nn.L1Loss()
    total_loss = total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, true_edges, predicted_edges = model(images)
        loss = classification_loss(logits, labels) + edge_weight * edge_loss(predicted_edges, true_edges)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def fit(model, train_loader, val_loader, device, epochs=50, learning_rate=1e-3, edge_weight=0.7):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, edge_weight)
        validation = evaluate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch + 1, "train_loss": train_loss, **validation}
        history.append(record)
        print(f"Epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} | validation accuracy {validation['accuracy']:.2f}%")
    return history
