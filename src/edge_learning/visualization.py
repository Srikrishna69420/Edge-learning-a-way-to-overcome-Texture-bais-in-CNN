"""Visualization helpers for edge reconstruction."""

import matplotlib.pyplot as plt
import torch


def show_edge_reconstruction(model, images, device):
    model.eval()
    with torch.no_grad():
        _, true_edges, predicted_edges = model(images[:1].to(device))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(true_edges[0, 0].cpu(), cmap="gray")
    axes[0].set_title("Sobel edges")
    axes[1].imshow(predicted_edges[0, 0].cpu(), cmap="gray")
    axes[1].set_title("Reconstructed edges")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()
