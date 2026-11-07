# Edge learning: reducing texture bias in CNNs

An exploratory CIFAR-10 project that combines a texture-processing CNN with an edge/shape-processing CNN. Sobel filters produce an edge map; a second branch processes the edge representation, and a decoder is trained to reconstruct edges alongside the classification objective.

## Quick start

The current implementation is a single script, `Code.py`. Run it from the repository root so that CIFAR-10 is downloaded to `./data` and the three example images at the root can be found.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Code.py
```

A CUDA-capable GPU is recommended. The script automatically selects CUDA when available.

## Repository layout

- `Code.py` — original end-to-end model, data pipeline, training loop, evaluation, and prediction script.
- `images/` — figures and visual material associated with the project.
- `test_cat.png`, `test_dog.png`, `test_frog.png` — example prediction images used by the script.
- `docs/REPRODUCIBILITY.md` — setup, data, and implementation notes.
- `requirements.txt` — dependencies.

## Method

The script creates Sobel edge maps from input images, processes RGB texture and grayscale edge information in separate convolutional branches, combines the learned representations for CIFAR-10 classification, and uses a decoder with an edge reconstruction loss. The total training loss in the script is classification loss plus a weighted edge loss.

## Important limitations

This is the original experimental implementation, not a validated benchmark. The script currently performs its training and prediction work at import/run time rather than exposing a reusable command-line interface. The data split is created with `random_split` without an explicit generator seed. Record the environment and run configuration before comparing results.

The original `Code.py` is preserved unchanged in this cleanup branch.
