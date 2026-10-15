import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import trapezoid
from torch.nn import functional as F

from .core import Model, spectral_pair


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(payload["config"], payload["method"])
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload["config"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, cfg = load(args.checkpoint)
    reference, _ = load(args.reference)
    arrays = dict(np.load(args.data, allow_pickle=False))
    x, y = torch.tensor(arrays["x"]), torch.tensor(arrays["y"])
    records = []
    parameters = tuple(model.parameters())
    for scale in sorted(args.scales):
        correct = reference_correct = total = 0
        ratios = []
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
                model, batch.float(), min(scale, cfg["low_resolution"]), model.method
            )
            if model.method in ("erm", "multiscale"):
                continue
            gradient = torch.autograd.grad(
                applied, parameters, retain_graph=True, allow_unused=True
            )
            restricted = torch.autograd.grad(supported, parameters, allow_unused=True)
            unsupported_norm = sum(
                (
                    (a if a is not None else torch.zeros_like(p))
                    - (b if b is not None else torch.zeros_like(p))
                )
                .square()
                .sum()
                for a, b, p in zip(gradient, restricted, parameters)
            ).sqrt()
            norm = sum(a.square().sum() for a in gradient if a is not None).sqrt()
            ratios.append(float(unsupported_norm / norm.clamp_min(1e-12)))
        records.append(
            {
                "resolution": scale,
                "accuracy": correct / total,
                "reference_accuracy": reference_correct / total,
                "ucgr": float(np.mean(ratios)) if ratios else None,
            }
        )
    axis = np.asarray([r["resolution"] for r in records], float)
    deficit = np.asarray([max(0, r["reference_accuracy"] - r["accuracy"]) for r in records])
    area = (
        float(trapezoid(deficit, axis) / (axis[-1] - axis[0]))
        if len(axis) > 1
        else float(deficit[0])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"negative_transfer_area": area, "resolutions": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
