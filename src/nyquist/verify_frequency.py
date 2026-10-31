"""Verify oracle annotations against the independent analytic frequency generator."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import cli, core, oracle_reference
from .frequency_data import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = generate(instances=30)
    # Four exact signal modes each have magnitude .15*N^2/2; the disjoint
    # nuisance modes cannot enter the supplied task support.
    spectrum = np.fft.fft2(data["x"][:, 0])
    retained = np.abs(spectrum) * data["oracle_support"]
    observed = retained[retained > 1e-3]
    np.testing.assert_allclose(observed, .15 * 32**2 / 2, rtol=2e-6)
    assert len(observed) == 4 * len(data["x"])
    assert np.all(data["oracle_support"][:, 0, 0] == 0)
    np.savez(output / "raw.npz", **data)
    cli.prepare(output / "raw.npz", output / "data", 17)
    cfg = dict(seed=17, device="cpu", epochs=1, patience=1, batch_size=6,
               classes=3, backbone="cnn", learning_rate=.001, low_resolution=16, consistency_weight=.1)
    cli.train(cfg, output / "data", output / "model", "nyquist")
    report = cli.evaluate(output / "model/model.pt", output / "data/test.npz", output / "evaluation")
    sample = dict(np.load(output / "data/test.npz"))
    payload = torch.load(output / "model/model.pt", weights_only=True)
    model = core.Model(cfg, "nyquist").eval()
    model.load_state_dict(payload["state"])
    values = core.spectral_pair(model, torch.tensor(sample["x"]), 16, "oracle_support", sample["oracle_support"])
    assert torch.isfinite(values[2])
    (output / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch("sys.argv", ["oracle_reference", "--config", str(output / "config.json"),
                            "--data", str(output / "data"), "--output", str(output / "reference")]):
        oracle_reference.main()
    reference = cli.evaluate(output / "reference/oracle_reference/model.pt", output / "data/test.npz", output / "oracle_evaluation")
    model = core.Model(cfg, "oracle_reference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    training = dict(np.load(output / "data/train.npz"))
    mini = {key: torch.as_tensor(value[:4]) for key, value in training.items() if key in ("x", "y", "oracle_support")}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = core.loss(model, mini, cfg)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    optimizer.step()
    assert not torch.equal(before["encoder.0.weight"], model.encoder[0].weight)
    assert not torch.equal(before["head.weight"], model.head.weight)
    cli.infer(output / "reference/oracle_reference/model.pt", output / "data/test.npz", output / "oracle_inference.npz")
    np.testing.assert_allclose(np.load(output / "oracle_inference.npz")["logits"],
                               np.load(output / "oracle_evaluation/predictions.npz")["logits"], atol=1e-7, rtol=1e-6)
    report["analytic_task_mode_magnitude"] = float(observed.mean())
    report["oracle_consistency"] = float(values[2].detach())
    report["oracle_reference"] = reference
    report["oracle_reference_loss"] = float(loss.detach())
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
