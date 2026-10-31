# Nyquist-Valid Consistency

Implementation of **Cross-Scale Consistency on Common Nyquist Support**. This is an implementation of the formulation; the paper's original experiment code, exact backbone/configuration and result logs were not provided. Generated examples below test functionality only.

## Install and data

Use Python >=3.11 in an isolated environment; activate it, then install from the repository root.

```bash
python -m venv .venv
python -m pip install -e .
python -m nyquist.cifar --output data/raw.npz --seed 17
python -m nyquist prepare --source data/raw.npz --output data/prepared --seed 17
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS. The adapter uses the released [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) split, reserves a seeded stratified 10% of the training set for validation, and never fits preprocessing on the test set. Image values are RGB float32 in [0,1], `(N,3,H,W)`; labels are integers `(N,)`. Custom data can use the same NPZ schema with `group` IDs for duplicate/source images and optional `split` (0=train,1=validation,2=test). `prepare` checks that a source group never crosses a supplied split; without `split` it partitions groups 60/20/20. No corpus download or full CIFAR training was performed in this review.

## Train, infer and evaluate

```bash
python -m nyquist train --config configs/default.json --data data/prepared --output runs/main
python -m nyquist infer --checkpoint runs/main/nyquist/model.pt --data data/prepared/test.npz --output results/inference.npz
python -m nyquist evaluate --checkpoint runs/main/nyquist/model.pt --data data/prepared/test.npz --output results/main
python -m nyquist suite --config configs/default.json --data data/prepared --output runs/comparison
python -m nyquist.diagnostics --checkpoint runs/comparison/nyquist/model.pt --reference runs/comparison/multiscale/model.pt --data data/prepared/test.npz --scales 16 24 32 --output results/diagnostics.json
python -m nyquist plot --inputs runs/main/nyquist results/main --output figures
```

Training uses AdamW with cosine schedule and best validation-loss checkpoint, seed17, lr0.001, batch32, 100 epochs, patience15, low_resolution16 and consistency_weight0.1. These are explicit implementation choices. The default backbone is a torchvision ResNet-18 with a CIFAR 3x3 stride-one stem, no max pool, spatial features through layer3, and a 256-dimensional pooled classifier. It is trained from scratch without downloading weights. Set `backbone=cnn` for the original compact three-convolution smoke backbone; old checkpoints without this field retain that behavior. ResNet-18 is an explicit implementation choice because the manuscript does not identify its original backbone. BlurPool replaces every stride-two convolution, including residual downsample paths, with stride-one convolution followed by the fixed binomial filter. The task objective is the mean CE of both branches for multi-scale methods, plus the applicable consistency term. ERM computes only the high-resolution branch. Inference uses the unchanged single encoder and classifier; `infer` requires only x and runs on CPU.

## Method, metric and entry mapping

| Paper module/endpoint | Code | Command/configuration |
| --- | --- | --- |
| Common feature support and alias-risk reliability | `core.spectral_pair` | `nyquist`, `low_resolution`, `consistency_weight` |
| Single-resolution and paired task losses | `core.loss` | `erm`, `multiscale` |
| Full, fixed low-pass and alias-only controls | `core.loss/spectral_pair` | `--methods full_consistency fixed_lowpass alias_only` |
| BlurPool + full consistency | `core.BlurPool`, `Model` | `--methods blurpool` |
| Accuracy, balanced accuracy, macro-F1, top3, ECE, NLL | `vision.classification_metrics`, `core.metrics` | evaluate; high and low resolution separately |
| UCGR and resolution curve | `diagnostics.unsupported_gradient_ratio`, `main` | diagnostics with a multi-scale reference checkpoint |
| Negative-transfer area | `diagnostics.negative_transfer_area` | diagnostics; fraction and percentage-point versions |
| Training throughput and cost | `cli.train`, `history.jsonl` | `training_seconds`, `training_images_per_second`, total epoch time |
| Parameters, conv/linear MACs, CPU/GPU latency and RSS | `cli.evaluate` | evaluate; FFT and resampling cost excluded from MACs |

Views use antialiased bilinear image downsampling. Lower-resolution features are bilinearly placed on the high-resolution grid before orthonormal FFT. The binary common mask retains integer cycles strictly below half the low feature-grid size on each axis; it is shared across channels. The fixed low-pass control uses one quarter of the high feature-grid size, independent of view ratio. For an exact half-scale pair its cutoff coincides with common support; reliability weighting still distinguishes it from the proposed loss. Alias risk is the fraction of input FFT energy outside the lower image-grid Nyquist support, and the detached reliability is `1-risk`. This is a declared energy-proxy choice, not an estimate of actual semantic loss or an original sampling operator's transfer function. BlurPool replaces learned stride-two subsampling with a fixed 3x3 binomial blur followed by decimation, then applies full consistency.

UCGR is `||grad(applied)-grad(applied restricted to common support)|| / ||grad(applied)||`, with zero denominator mapped to zero. In particular a binary projected objective has zero unsupported gradient up to numerical error; this algebraic result is not evidence for the nonzero paper measurement. Fixed-mask diagnostics restrict the actual applied mask by common support rather than incorrectly subtracting the same gradient from itself. Per-batch parameter-gradient norm ratios are weighted by sample count; classifier-head gradients are absent from this consistency diagnostic. At each stress resolution, the paired low resolution preserves the configured training view ratio. The reported NTA integrates positive accuracy deficits relative to the supplied multi-scale reference uniformly over the supplied resolution axis, divides by axis range, and also reports 100x the fraction as percentage points. This convention is explicit because the paper does not provide the original aggregation code.

All seven methods share the classifier parameterization and optimization defaults. The evaluation-only oracle accepts binary task-frequency annotations through `--oracle-support support.npy` in diagnostics, or an `oracle_support` array in the evaluation NPZ. Masks are unshifted FFT grids of shape `(H,W)` or `(N,H,W)`; integer cycles are mapped onto the feature grid. Diagnostics report oracle-masked consistency and its common-support UCGR. The seven deployable training methods never consume these annotations. A separately invoked privileged oracle-reference experiment is described below; it is excluded from the normal train/suite method choices. For natural CIFAR images the exact task-frequency support is not given by a class label; supply independent ground-truth annotations when available. The mask-shape fixture tests empty/DC-only supports without claiming they are CIFAR semantic truth. The separate controlled frequency family below provides analytically known input task bands. The present code does not claim to reproduce the paper's reported numerical table.

## Outputs and actual verification

Training writes `model.pt` and `history.jsonl`; evaluation writes `predictions.npz` and `metrics.json`; diagnostics writes a resolution/UCGR/NTA JSON. Plotting reads training histories and evaluation metric directories.

```bash
python -m nyquist.verify --output runs/check
```

The functional check uses generated 32x32 images with explicit disjoint splits, all seven methods, one epoch, finite loss, nonzero gradients, parameter updates, save/load agreement (atol1e-7, rtol1e-6), x-only inference agreement, actual evaluation/diagnostics and figures. It checks hand examples UCGR=0.8, NTA=0.1, ECE=0.25 and NLL=-log(0.56)/2, including a nonshared fixed-mask regression. Tested with Python3.13.5, torch2.7.1+cu118 and numpy2.1.3, CPU1 thread, seed17. The tested Anaconda environment needs process-local `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL` to avoid duplicate OpenMP runtime loading. No shared packages or global settings are changed.


## Standard-backbone and oracle verification

```bash
python -m nyquist.verify_extensions --output runs/extensions
python -m nyquist.diagnostics --checkpoint runs/main/nyquist/model.pt --reference runs/comparison/multiscale/model.pt --data data/prepared/test.npz --scales 16 24 32 --oracle-support data/task_support.npy --output results/oracle_diagnostics.json
```

`verify_extensions` additionally exercises all seven methods on standard ResNet-18, checking gradients and updates in the encoder and classifier, tensor-only save/load agreement, and an actual train/evaluate/infer/diagnostic/plot chain. The CIFAR adapter is tested with generated files in the original distributed pickle schema; only the fixture's official-download MD5 checks are mocked, because generated fixtures necessarily differ from the release checksums. Image IDs remain disjoint across train/validation/test. Empty/DC-only/shared/per-example oracle masks and training isolation are checked. The annotations never alter a deployable training-method loss. The original `verify` entry continues to test all seven methods on the compact CPU backbone.


## Known-spectrum oracle reference

```bash
python -m nyquist.frequency_data --output data/frequency.npz --instances 300 --size 32 --classes 3 --seed 17
python -m nyquist prepare --source data/frequency.npz --output data/frequency --seed 17
python -m nyquist.oracle_reference --config configs/frequency.json --data data/frequency --output runs/oracle
python -m nyquist evaluate --checkpoint runs/oracle/oracle_reference/model.pt --data data/frequency/test.npz --output results/oracle
python -m nyquist.verify_frequency --output runs/frequency-check
```

`frequency_data` encodes a label in one of three analytic sinusoidal signal frequencies, adds an independent high-frequency nuisance, and exports the four exact task Fourier bins per image. These controlled data are not CIFAR performance evidence. The hand check independently verifies their FFT magnitude `.15*32^2/2=76.8` and excludes the nuisance/DC bins. Source IDs and seeded stratified splits are disjoint.

The manuscript identifies the oracle as an evaluation-only reference without supplying its training recipe. Two explicitly separate uses are provided: `diagnostics` audits supplied supports without training, while `oracle_reference` evaluates the effect of privileged training/validation task supports using the same classifier/objective. This reference is not a deployable method and is excluded from the main suite. Its fitting code does not read the test file; test masks/labels never fit any checkpoint or threshold. Both references require genuine independently supplied task-band annotations and cannot infer semantic truth from a CIFAR class label. Actual oracle-reference training, encoder/head gradients and updates, checkpoint evaluation and x-only inference were checked on the controlled family; the paper's table is not reproduced.
