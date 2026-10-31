"""Fetch a small real CIFAR-100 subset from the public Hugging Face mirror."""

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


DATASET = "uoft-cs/cifar100"


def rows(split, count):
    """Read at most 100 public source rows, preserving their official split IDs."""
    import requests

    response = requests.get("https://datasets-server.huggingface.co/rows", params={
        "dataset": DATASET, "config": "cifar100", "split": split, "offset": 0, "length": count}, timeout=30)
    response.raise_for_status()
    result = response.json()["rows"]
    if len(result) != count:
        raise ValueError("Dataset viewer returned fewer source rows than requested")
    return result


def image(row):
    """Decode the source image without storing expiring public CDN URLs."""
    import requests

    response = requests.get(row["row"]["img"]["src"], timeout=30)
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as source:
        pixels = np.array(source.convert("RGB"), copy=True)
    if pixels.shape != (32, 32, 3):
        raise ValueError("Expected a 32x32 RGB CIFAR image")
    return pixels.transpose(2, 0, 1).astype(np.float32) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 2 <= args.train_count <= 100 or not 1 <= args.test_count <= 100:
        parser.error("This functional subset tool requires 2..100 training and 1..100 test rows")
    if not 0 < args.validation_count < args.train_count:
        parser.error("Validation rows must be a proper nonempty subset of the source training rows")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    training, testing = rows("train", args.train_count), rows("test", args.test_count)
    all_rows = training + testing
    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = np.stack(list(pool.map(image, all_rows)))
    original_row = np.array([row["row_idx"] for row in all_rows], dtype=np.int64)
    source_split = np.r_[np.zeros(len(training), dtype=np.int64), np.ones(len(testing), dtype=np.int64)]
    split = source_split * 2
    validation = np.random.default_rng(args.seed).permutation(len(training))[:args.validation_count]
    split[validation] = 1
    fine = np.array([row["row"]["fine_label"] for row in all_rows], dtype=np.int64)
    coarse = np.array([row["row"]["coarse_label"] for row in all_rows], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=pixels, y=fine, fine_label=fine, coarse_label=coarse,
                        group=original_row + 50000 * source_split, source_row=original_row,
                        source_split=source_split, split=split)
    print(json.dumps(dict(dataset=DATASET, source_train_rows=len(training), source_test_rows=len(testing),
                          split_rows={role: int(np.sum(split == i)) for i, role in enumerate(("train", "validation", "test"))},
                          functional_only=True)))


if __name__ == "__main__":
    main()
