# Edge learning: reducing texture bias in CNNs

An exploratory CIFAR-10 project combining an RGB texture branch with a Sobel-edge shape branch. A decoder is trained to reconstruct the edge representation alongside the classification objective.

## Modular implementation (recommended)

The refactored package is in `src/edge_learning/`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m edge_learning.cli --epochs 50
```

The CLI downloads CIFAR-10 to `./data`, trains the model, evaluates the held-out test split, saves a checkpoint to `artifacts/edge_texture_cnn.pt`, and displays an edge reconstruction plus predictions for the bundled cat/dog/frog examples. Use `--skip-examples` on headless machines. Run `python -m edge_learning.cli --help` for options.

## Project structure

- `src/edge_learning/data.py` — CIFAR-10 transforms and reproducible 80/10/10 split.
- `src/edge_learning/model.py` — dual-branch CNN and edge decoder.
- `src/edge_learning/engine.py` — training and evaluation loops.
- `src/edge_learning/predict.py` — prediction for external images.
- `src/edge_learning/visualization.py` — edge reconstruction plot.
- `src/edge_learning/cli.py` — command-line entry point.
- `tests/` — lightweight architecture and gradient tests.
- `Code.py` — original single-file experiment, retained unchanged.
- `images/` — project figures; `test_*.png` — sample prediction images.

## Model idea

Sobel filters generate an edge magnitude map. One CNN branch processes RGB texture; another processes the edge map. The learned representations are combined for CIFAR-10 classification, while a decoder reconstructs the edge map. The objective is cross-entropy plus a weighted L1 reconstruction loss.

## Reproducibility and differences from the original

The modular version makes the split reproducible (seed 42 by default) and uses deterministic preprocessing for validation/test data. The original script applies random crop/flip transforms to every split and has no explicit split seed. This is an intentional experimental hygiene improvement, so its metrics need not match the original script. The original `Code.py` remains available for exact historical reference.

## Tests

```bash
python -m unittest discover -s tests
```

Tests check model output shapes and gradient flow; they do not train on the full CIFAR-10 dataset or establish research performance.
