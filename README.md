# Nyquist

Implementation of **Cross-scale consistency on common Nyquist support**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The CIFAR adapter downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves 10% of training images for validation and preserves the released test set. Preparation retains original image IDs; the low-resolution branch is generated online.

```bash
python -m nyquist.cifar --output data/raw.npz --seed 17
python -m nyquist prepare --source data/raw.npz --output data/prepared --seed 17
```

## Training

Use the same `configs/default.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m nyquist train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m nyquist evaluate --checkpoint runs/main/nyquist/model.pt --data data/prepared/test.npz --output results/nyquist
```

## Comparisons and ablations

`suite` trains the implemented methods on the same prepared split. Use `--methods` to select a subset, then evaluate each method checkpoint with the evaluation command above.

```bash
python -m nyquist suite --config configs/default.json --data data/prepared --output runs/comparison
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m nyquist plot --inputs runs/main/nyquist results/nyquist --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
