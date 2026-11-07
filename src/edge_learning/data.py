"""CIFAR-10 data pipeline with separate train and evaluation transforms."""

from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def build_loaders(root="./data", batch_size=128, num_workers=0, seed=42, download=True):
    """Return train/validation/test loaders using an 80/10/10 split.

    Validation and test examples use deterministic preprocessing. The original
    single-script version applied random augmentation to all three subsets.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    augmented = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    evaluation = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=eval_transform)
    total = len(augmented)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    generator = __import__("torch").Generator().manual_seed(seed)
    indices = __import__("torch").randperm(total, generator=generator).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    train_loader = DataLoader(Subset(augmented, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(evaluation, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(evaluation, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
