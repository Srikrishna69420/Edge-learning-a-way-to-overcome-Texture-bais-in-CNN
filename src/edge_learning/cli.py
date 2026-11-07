"""Command-line entry point for training and evaluation."""

import argparse
from pathlib import Path
import torch
from .data import build_loaders
from .engine import evaluate, fit
from .model import EdgeTextureCNN
from .predict import predict_image
from .visualization import show_edge_reconstruction


def main():
    parser = argparse.ArgumentParser(description="Train the texture/edge CIFAR-10 CNN")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--edge-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default="artifacts/edge_texture_cnn.pt")
    parser.add_argument("--skip-examples", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers, args.seed)
    model = EdgeTextureCNN().to(device)
    fit(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.edge_weight)
    print("Test metrics:", evaluate(model, test_loader, device))

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": model.classes, "seed": args.seed}, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")

    if not args.skip_examples:
        show_edge_reconstruction(model, next(iter(test_loader))[0], device)
        for name in ("test_cat.png", "test_dog.png", "test_frog.png"):
            path = Path(name)
            if path.is_file():
                print(f"{name}: {predict_image(model, path, device)}")


if __name__ == "__main__":
    main()
