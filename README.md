# Nyquist-Valid Consistency

Cross-scale classification with feature consistency restricted to common Nyquist support.

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

Training defaults: AdamW, cosine schedule, best validation-loss checkpoint, seed 17, learning rate 0.001, batch size 32, 100 epochs, patience 15, `low_resolution=16`, and `consistency_weight=0.1`. The default ResNet-18 has a CIFAR 3x3 stride-one stem, no max pool, layer3 spatial features and a 256-dimensional pooled classifier, initialized from scratch. `backbone=cnn` selects a compact three-convolution model; checkpoints without the field use that model. BlurPool replaces stride-two convolutions, including residual downsample paths, with stride-one convolutions and a fixed binomial filter. Multi-scale methods average the two branches' CE losses and add their consistency term; ERM uses the high-resolution branch. Inference uses one encoder/classifier and requires only `x`.

## Method, metric and entry mapping

Views use antialiased bilinear downsampling. Low-resolution features are bilinearly placed on the high-resolution grid before orthonormal FFT. The shared binary mask keeps integer cycles strictly below half the low feature-grid size on each axis. The fixed low-pass control uses one quarter of the high feature-grid size. Alias risk is the input FFT energy fraction outside the lower image-grid Nyquist support; detached reliability is `1-risk`. BlurPool uses a fixed 3x3 binomial blur and decimation followed by full consistency.

UCGR is `||grad(applied)-grad(applied restricted to common support)|| / ||grad(applied)||`, with zero denominator mapped to zero. A binary projected objective has zero unsupported gradient up to numerical error. Fixed-mask diagnostics intersect the applied mask with common support. Batch gradient ratios are sample-weighted and exclude classifier-head parameters. Stress resolutions preserve the configured view ratio. NTA integrates positive accuracy deficits relative to the supplied multi-scale reference, divides by the resolution-axis range, and reports both fractions and percentage points.

All seven methods share classifier and optimization defaults. Optional oracle diagnostics require independently supplied binary task-frequency annotations via `--oracle-support support.npy` or evaluation-array `oracle_support`. Masks are unshifted FFT grids `(H,W)` or `(N,H,W)`; integer cycles map onto the feature grid. Deployable training methods use no oracle annotations. The controlled frequency family below supplies analytic task bands; natural-image class labels alone do not specify such bands.

## Outputs and CPU checks

Training writes `model.pt` and `history.jsonl`; evaluation writes `predictions.npz` and `metrics.json`; diagnostics writes a resolution/UCGR/NTA JSON. Plotting reads training histories and evaluation metric directories.

```bash
python -m nyquist.verify --output runs/check
```

For CPU checks in the Anaconda environment, use process-local `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `MKL_THREADING_LAYER=SEQUENTIAL`.

## Backbone and oracle checks

```bash
python -m nyquist.verify_extensions --output runs/extensions
python -m nyquist.diagnostics --checkpoint runs/main/nyquist/model.pt --reference runs/comparison/multiscale/model.pt --data data/prepared/test.npz --scales 16 24 32 --oracle-support data/task_support.npy --output results/oracle_diagnostics.json
```

`verify_extensions` checks the seven ResNet-18 branches, checkpoint restoration, CIFAR-format loading, disjoint source IDs and oracle-mask isolation. `verify` uses the compact CPU backbone.

## Small real-CIFAR check

```bash
python -m pip install -e ".[remote]"
python -m nyquist.cifar_subset --train-count 40 --validation-count 8 --test-count 10 --output data/cifar-small.npz
python -m nyquist prepare --source data/cifar-small.npz --output data/cifar-small --seed 17
python -m nyquist suite --config configs/data_smoke.json --data data/cifar-small --output runs/real-cifar-check
```

This optional tool fetches images and fine/coarse labels from the public [uoft-cs/cifar100 mirror](https://huggingface.co/datasets/uoft-cs/cifar100) through the dataset viewer. It preserves source row IDs and train/test membership, reserves validation from source training rows with seed 17, and caps each source split at 100 rows. Images use the viewer's encoding. Use `cifar` for the Toronto archive adapter with its release checksum checks. Task-frequency annotations are supplied independently.

## Known-spectrum oracle reference

```bash
python -m nyquist.frequency_data --output data/frequency.npz --instances 300 --size 32 --classes 3 --seed 17
python -m nyquist prepare --source data/frequency.npz --output data/frequency --seed 17
python -m nyquist.oracle_reference --config configs/frequency.json --data data/frequency --output runs/oracle
python -m nyquist evaluate --checkpoint runs/oracle/oracle_reference/model.pt --data data/frequency/test.npz --output results/oracle
python -m nyquist.verify_frequency --output runs/frequency-check
```

`frequency_data` encodes labels in three analytic sinusoidal signal frequencies, adds independent high-frequency nuisance, and exports the four task Fourier bins per image. Source IDs and seeded stratified splits are disjoint.

`diagnostics` audits supplied task supports without training. `oracle_reference` trains with privileged training/validation supports using the same classifier/objective and is invoked separately from the normal suite. It fits no test file or test masks. Both paths require independent task-band annotations.
