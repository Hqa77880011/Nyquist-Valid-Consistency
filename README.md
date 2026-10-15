# Nyquist-Valid-Consistency

Implementation of **Cross-scale consistency on common Nyquist support**.

Projects paired feature spectra onto jointly representable frequencies and weights them by input alias energy. Spectral operations are used during training; inference uses the shared encoder.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/Nyquist-Valid-Consistency.git
cd Nyquist-Valid-Consistency
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) supplies the classification images. Preserve the official test set and reserve a training holdout for selection.

`x` is float32 RGB [N,3,32,32], `y` a fine class ID and group the original image ID. The low-resolution branch is generated online with anti-aliased interpolation.

```bash
python -m nyquist prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m nyquist train --config configs/default.json --data data/prepared --output runs/main
python -m nyquist suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. nyquist, erm, multiscale, full_consistency, fixed_lowpass and alias_only.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m nyquist evaluate --checkpoint runs/main/nyquist/model.pt --data data/prepared/test.npz --output results/nyquist
python -m nyquist plot --inputs runs/main/nyquist results/nyquist --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

High/low resolution accuracy, macro F1, ECE/NLL and resolution gap. The diagnostic module measures spectral gradient attribution and negative-transfer area against a supplied multi-scale reference.

## Implementation conventions

The matched encoder is a compact convolutional backbone. Common support is measured on the feature-grid Nyquist limits; alias energy is measured on the input grid. Low-resolution features are bilinearly placed on the high-resolution coordinate grid before FFT. Use the same normalization and optimizer budget for every method.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m nyquist.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m nyquist.cifar --output data/raw.npz
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

## Mechanism diagnostics

```bash
python -m nyquist.diagnostics --checkpoint runs/main/nyquist/model.pt --reference runs/comparison/multiscale/model.pt --data data/prepared/test.npz --scales 16 24 32 --output results/diagnostics.json
```

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
