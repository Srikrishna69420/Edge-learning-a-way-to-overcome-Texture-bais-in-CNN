# Reproducibility and implementation notes

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python Code.py
```

The script downloads CIFAR-10 using torchvision. It also attempts predictions on `test_cat.png`, `test_dog.png`, and `test_frog.png`, so keep the working directory at the repository root.

## Dependencies

The implementation imports PyTorch, torchvision, Matplotlib, Pillow, and the Python standard library. The versions in `requirements.txt` are minimum versions, not a hardware-specific CUDA lockfile. Install a PyTorch build compatible with your machine.

## Data and split

CIFAR-10 is downloaded into `./data`. The script creates train/validation/test subsets from the training dataset using an 80/10/10 split. It does not set a generator for `random_split`, so the exact split is not guaranteed to repeat between runs.

## Model outline

- Sobel X/Y kernels generate an edge magnitude map.
- A texture branch processes the RGB image.
- A shape branch processes the edge map.
- A decoder reconstructs the edge representation.
- The classifier combines the branch features.
- Training minimizes cross-entropy plus a weighted L1 edge-reconstruction loss.

## Record for future experiments

Record Python/PyTorch/torchvision versions, CUDA runtime, GPU model, random seed, data split, batch size, optimizer settings, epoch count, and evaluation results. The script currently hard-codes most of these settings.

## Known scope

This documentation does not claim that the experiment has been re-run or independently validated. The implementation is still a single script; modular extraction and automated tests are separate future refactoring work.
