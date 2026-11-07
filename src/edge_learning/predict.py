"""Prediction helper for user-supplied RGB images."""

from PIL import Image
import torch
from torchvision import transforms
from .data import CIFAR_MEAN, CIFAR_STD
from .model import EdgeTextureCNN


def predict_image(model, path, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(image)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax())
    return EdgeTextureCNN.classes[index], float(probabilities[index])
