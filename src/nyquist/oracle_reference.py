"""Privileged task-support reference, separate from deployable method training."""

import argparse
import json
from pathlib import Path

from .cli import Arrays, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # The reference consumes annotations only from training/validation. It does
    # not load test images, labels or masks while fitting a checkpoint.
    for split in ("train", "validation"):
        source = Arrays(Path(args.data) / (split + ".npz"))
        if "oracle_support" not in source.data:
            raise ValueError(f"{split} requires independent oracle_support annotations")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train(cfg, args.data, Path(args.output) / "oracle_reference", "oracle_reference")


if __name__ == "__main__":
    main()
