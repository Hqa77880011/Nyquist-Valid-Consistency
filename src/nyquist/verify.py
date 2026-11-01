"""CPU checks for cross-scale consistency on generated images."""

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from . import cli, core, diagnostics
from .vision import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    np.savez(output / "raw.npz", x=rng.random((30, 3, 32, 32), dtype=np.float32),
             y=np.tile(np.arange(3), 10), group=np.arange(30),
             split=np.repeat([0, 1, 2], [18, 6, 6]))
    cli.prepare(output / "raw.npz", output / "data", 17)
    cfg = json.loads((Path(__file__).parents[2] / "configs/default.json").read_text())
    cfg.update(classes=3, epochs=1, batch_size=6, backbone="cnn")
    batch = next(iter(torch.utils.data.DataLoader(cli.Arrays(output / "data/train.npz"), batch_size=6)))
    summary = {}
    for method in core.METHODS:
        torch.manual_seed(17)
        model = core.Model(cfg, method)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        value = core.loss(model, batch, cfg)
        assert torch.isfinite(value)
        value.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), (method, name)
            assert parameter.grad.abs().sum() > 0, (method, name)
        optimizer.step()
        assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
        model.eval()
        with torch.no_grad():
            expected = model(batch["x"])[0]
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = core.Model(cfg, method).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(batch["x"])[0], expected, atol=1e-7, rtol=1e-6)
        cli.train(cfg, output / "data", output / method, method)
        report = cli.evaluate(output / method / "model.pt", output / "data/test.npz", output / "evaluation" / method)
        summary[method] = {"initial_loss": float(value.detach()), "accuracy": report["accuracy"]}
    probe = torch.tensor([3., 4.], requires_grad=True)
    np.testing.assert_allclose(diagnostics.unsupported_gradient_ratio(probe.square().sum() / 2,
                                                                    probe[0].square() / 2, (probe,)), 0.8)
    np.testing.assert_allclose(diagnostics.negative_transfer_area([16, 32], [0.4, 0.8], [0.6, 0.8]), 0.1)
    # With a 32->8 pair the fixed mask includes nonshared bands: UCGR must not be identically zero.
    model = core.Model(cfg, "fixed_lowpass")
    values = core.spectral_pair(model, batch["x"], 8, "fixed_lowpass")
    ratio = diagnostics.unsupported_gradient_ratio(values[2], values[5], tuple(model.parameters()))
    assert ratio > 0.001, ratio
    metric = classification_metrics(np.log(np.array([[0.8, 0.2], [0.3, 0.7]])), np.array([0, 1]))
    np.testing.assert_allclose(metric["accuracy"], 1)
    np.testing.assert_allclose(metric["ece"], 0.25)
    np.testing.assert_allclose(metric["nll"], -np.log(0.56) / 2)
    with patch("sys.argv", ["diagnostics", "--checkpoint", str(output / "nyquist/model.pt"),
                            "--reference", str(output / "multiscale/model.pt"), "--data", str(output / "data/test.npz"),
                            "--scales", "16", "32", "--output", str(output / "diagnostics.json")]):
        diagnostics.main()
    cli.plot([output / "nyquist", output / "evaluation/nyquist"], output / "figures")
    with np.load(output / "data/test.npz") as source:
        np.savez(output / "unlabeled.npz", x=source["x"])
    cli.infer(output / "nyquist/model.pt", output / "unlabeled.npz", output / "inference.npz")
    np.testing.assert_allclose(np.load(output / "inference.npz")["logits"],
                               np.load(output / "evaluation/nyquist/predictions.npz")["logits"], atol=1e-7, rtol=1e-6)
    summary.update(seed=17, fixed_mask_ucgr=ratio, metric_fixture=metric,
                   torch=torch.__version__, numpy=np.__version__)
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
