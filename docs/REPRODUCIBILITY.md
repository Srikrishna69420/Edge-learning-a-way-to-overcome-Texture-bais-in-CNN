# Implementation and reproducibility notes

## Recommended entry point

Install the package in editable mode and run `python -m edge_learning.cli`. The package uses a `src/` layout and exposes model, data, training, evaluation, prediction, and visualization modules separately.

## Dataset and split

CIFAR-10 is downloaded through torchvision into `./data`. The modular loader creates an 80/10/10 split using a seeded permutation. Training uses random horizontal flips and random crops; validation and test use normalization only. This differs from the legacy `Code.py`, which applies the same random transform to all subsets and uses an unseeded `random_split`.

## Architecture

- Fixed Sobel X/Y kernels are registered as model buffers, so they follow the model across CPU/GPU.
- RGB texture branch: 3→64→128→256 channels, with batch normalization, ReLU, and 2× pooling after each block.
- Edge branch: 1→64→128→256 channels with corresponding pooling.
- Decoder reconstructs a 32×32 edge map from intermediate shape features using a skip connection.
- Classifier concatenates flattened texture features with half-scaled shape features.
- Objective: cross-entropy + `edge_weight × L1(reconstructed_edges, sobel_edges)`.

## Training and artifacts

Default settings mirror the original script's 50 epochs, batch size 128, Adam learning rate 0.001, StepLR step size 20 and gamma 0.5, and edge loss weight 0.7. The CLI saves a state-dict checkpoint under `artifacts/`. Checkpoints are ignored by Git by default.

## Reproducibility checklist

Record Python, PyTorch, torchvision, CUDA, GPU, seed, batch size, learning rate, epoch count, split indices/seed, and edge-loss weight. The default split seed is 42. GPU kernels may still be nondeterministic.

## Legacy implementation

The root-level `Code.py` is preserved unchanged. It trains immediately when run and is retained as a historical snapshot. The modular package is the recommended entry point for future experiments. Do not assume its results are numerically identical to the legacy script because the data split and evaluation transforms are improved.
