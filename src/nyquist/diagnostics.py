import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import trapezoid
from torch.nn import functional as F

from .core import Model, spectral_pair


def unsupported_gradient_ratio(applied, supported, parameters):
    """Ratio of the unsupported consistency-gradient norm to the applied norm."""
    gradient = torch.autograd.grad(applied, parameters, retain_graph=True, allow_unused=True)
    restricted = torch.autograd.grad(supported, parameters, allow_unused=True)
    numerator = sum(((a if a is not None else torch.zeros_like(p)) -
                     (b if b is not None else torch.zeros_like(p))).square().sum()
                    for a, b, p in zip(gradient, restricted, parameters)).sqrt()
    denominator = sum(a.square().sum() for a in gradient if a is not None).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).detach())


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(payload["config"], payload["method"])
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload["config"]


def negative_transfer_area(resolutions, accuracy, reference_accuracy):
    """Average positive accuracy deficit, integrated uniformly over resolution."""
    axis = np.asarray(resolutions, float)
    deficit = np.maximum(0, np.asarray(reference_accuracy) - np.asarray(accuracy))
    return (float(trapezoid(deficit, axis) / (axis[-1] - axis[0]))
            if len(axis) > 1 else float(deficit[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-support", help="Evaluation-only binary .npy support in unshifted FFT order")
    args = parser.parse_args()
    if len(set(args.scales)) != len(args.scales) or any(scale < 4 for scale in args.scales):
        parser.error("scales must be unique and >=4")
    model, cfg = load(args.checkpoint)
    reference, _ = load(args.reference)
    with np.load(args.data, allow_pickle=False) as source:
        arrays = dict(source)
    x, y = torch.tensor(arrays["x"]), torch.tensor(arrays["y"])
    oracle_support = np.load(args.oracle_support, allow_pickle=False) if args.oracle_support else arrays.get("oracle_support")
    if oracle_support is not None and oracle_support.ndim == 3 and len(oracle_support) != len(x):
        parser.error("Per-example oracle support must match the evaluation sample count")
    records = []
    parameters = tuple(model.parameters())
    for scale in sorted(args.scales):
        correct = reference_correct = total = 0
        ratios = []
        oracle_ratios = []
        oracle_losses = []
        paired_scale = max(4, round(scale * cfg["low_resolution"] / x.shape[-1]))
        for offset in range(0, len(x), cfg["batch_size"]):
            batch = x[offset : offset + cfg["batch_size"]]
            labels = y[offset : offset + cfg["batch_size"]]
            resized = F.interpolate(
                batch.float(), (scale, scale), mode="bilinear", align_corners=False, antialias=True
            )
            with torch.no_grad():
                predicted, _ = model(resized)
                baseline, _ = reference(resized)
                correct += int((predicted.argmax(-1) == labels).sum())
                reference_correct += int((baseline.argmax(-1) == labels).sum())
                total += len(labels)
            _, _, applied, _, _, supported = spectral_pair(
                model, resized, paired_scale, model.method
            )
            if model.method in ("erm", "multiscale"):
                continue
            ratios.append((unsupported_gradient_ratio(applied, supported, parameters), len(labels)))
            if oracle_support is not None:
                annotation = oracle_support if oracle_support.ndim == 2 else oracle_support[offset:offset + len(batch)]
                _, _, oracle_loss, _, _, common_loss = spectral_pair(
                    model, resized, paired_scale, "oracle_support", annotation
                )
                oracle_losses.append((float(oracle_loss.detach()), len(batch)))
                oracle_ratios.append((unsupported_gradient_ratio(oracle_loss, common_loss, parameters), len(batch)))
        records.append(
            {
                "resolution": scale,
                "paired_low_resolution": paired_scale,
                "accuracy": correct / total,
                "reference_accuracy": reference_correct / total,
                "ucgr": float(np.average([value for value, _ in ratios], weights=[count for _, count in ratios])) if ratios else None,
                "oracle_support_ucgr": float(np.average([v for v, _ in oracle_ratios], weights=[n for _, n in oracle_ratios])) if oracle_ratios else None,
                "oracle_support_consistency": float(np.average([v for v, _ in oracle_losses], weights=[n for _, n in oracle_losses])) if oracle_losses else None,
            }
        )
    axis = np.asarray([r["resolution"] for r in records], float)
    area = negative_transfer_area(axis, [r["accuracy"] for r in records],
                                  [r["reference_accuracy"] for r in records])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"negative_transfer_area": area, "negative_transfer_area_percentage_points": 100 * area, "resolutions": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
