"""Controlled frequency classification with a known evaluation-only task spectrum."""

import argparse
from pathlib import Path

import numpy as np


def generate(instances=300, size=32, classes=3, seed=17):
    """Generate labels determined by a signal frequency, with independent nuisance bands."""
    if classes < 2 or size < 4 * (classes + 1) or instances < 5 * classes:
        raise ValueError("Require classes>=2, size>=4*(classes+1), and at least five objects per class")
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size) / size, np.arange(size) / size, indexing="ij")
    images, labels, masks = [], [], []
    for index in range(instances):
        label = index % classes
        frequency = label + 1
        phase = rng.uniform(0, 2 * np.pi, 3)
        signal = .15 * (np.cos(2 * np.pi * frequency * xx + phase[0]) +
                         np.sin(2 * np.pi * frequency * yy + phase[1]))
        nuisance_frequency = int(rng.integers(size // 3, size // 2))
        nuisance = .1 * np.cos(2 * np.pi * nuisance_frequency * (xx + yy) + phase[2])
        image = .5 + signal + nuisance
        support = np.zeros((size, size), dtype=np.uint8)
        support[0, [frequency, -frequency]] = 1
        support[[frequency, -frequency], 0] = 1
        images.append(np.repeat(image[None], 3, axis=0))
        labels.append(label)
        masks.append(support)
    labels = np.asarray(labels, dtype=np.int64)
    split = np.full(instances, 2, dtype=np.int64)
    for label in range(classes):
        ids = rng.permutation(np.flatnonzero(labels == label))
        first, second = max(1, int(.6 * len(ids))), min(len(ids) - 1, int(.8 * len(ids)))
        split[ids[:first]] = 0
        split[ids[first:second]] = 1
    return dict(x=np.asarray(images, dtype=np.float32), y=labels,
                oracle_support=np.asarray(masks), group=np.arange(instances), split=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instances", type=int, default=300)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **generate(args.instances, args.size, args.classes, args.seed))


if __name__ == "__main__":
    main()
