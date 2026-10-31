"""Exercise the standard backbone, real CIFAR file format and oracle-only audit."""

import argparse
import io
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import cifar, cli, core, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    folder = output / "cifar" / "cifar-100-python"
    folder.mkdir(parents=True)
    # Canonical distributed pickle schema; fixtures intentionally have different MD5s.
    for name, repeats in (("train", 2), ("test", 1)):
        labels = np.tile(np.arange(100), repeats)
        payload = dict(data=rng.integers(0, 256, (len(labels), 3072), dtype=np.uint8),
                       fine_labels=labels.tolist(), coarse_labels=(labels // 5).tolist())
        (folder / name).write_bytes(pickle.dumps(payload))
    (folder / "meta").write_bytes(pickle.dumps(dict(
        fine_label_names=[str(i) for i in range(100)],
        coarse_label_names=[str(i) for i in range(20)])))
    with patch("torchvision.datasets.cifar.check_integrity", return_value=True), patch(
        "sys.argv", ["cifar", "--root", str(output / "cifar"), "--output", str(output / "packed.npz")]
    ):
        cifar.main()
    cli.prepare(output / "packed.npz", output / "prepared", 17)
    packed = dict(np.load(output / "packed.npz"))
    assert (packed["split"] == 2).sum() == 100
    assert (packed["split"] == 1).sum() == 100
    batch = {"x": torch.from_numpy(packed["x"][:4]), "y": torch.tensor([0, 1, 2, 0])}
    cfg = json.loads((Path(__file__).parents[2] / "configs/default.json").read_text())
    cfg.update(classes=3, epochs=1, batch_size=4, backbone="resnet18")
    records = {}
    for method in core.METHODS:
        torch.manual_seed(17)
        model = core.Model(cfg, method)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        value = core.loss(model, batch, cfg)
        assert torch.isfinite(value)
        value.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (method, name)
            assert p.grad.abs().sum() > 0, (method, name)
        optimizer.step()
        assert not torch.equal(before["encoder.0.weight"], model.encoder[0].weight)
        assert not torch.equal(before["head.weight"], model.head.weight)
        model.eval()
        with torch.no_grad():
            expected = model(batch["x"])[0]
            # Oracle annotations cannot affect any training-method objective.
            a = core.loss(model, batch, cfg)
            b = core.loss(model, dict(batch, oracle_support=torch.ones(4, 32, 32)), cfg)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = core.Model(cfg, method).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(batch["x"])[0], expected, atol=1e-7, rtol=1e-6)
        records[method] = {"loss": float(value.detach()), "reload_atol": 1e-7, "reload_rtol": 1e-6}
    # A known task-frequency annotation retains only DC. An empty oracle support
    # must give exactly zero consistency and zero unsupported gradient.
    model = core.Model(cfg, "nyquist").eval()
    empty = np.zeros((32, 32), dtype=np.float32)
    values = core.spectral_pair(model, batch["x"], 16, "oracle_support", empty)
    np.testing.assert_allclose(float(values[2].detach()), 0)
    np.testing.assert_allclose(diagnostics.unsupported_gradient_ratio(values[2], values[5], tuple(model.parameters())), 0)
    dc = empty.copy()
    dc[0, 0] = 1
    per_example = np.broadcast_to(dc, (4, 32, 32)).copy()
    a = core.spectral_pair(model, batch["x"], 16, "oracle_support", dc)[2]
    b = core.spectral_pair(model, batch["x"], 16, "oracle_support", per_example)[2]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Exercise the public train/evaluate/infer/diagnostics handlers with ResNet.
    for index, split in enumerate(("train", "validation", "test")):
        selected = np.flatnonzero(packed["split"] == index)[:4]
        np.savez(output / "prepared" / (split + ".npz"), x=packed["x"][selected],
                 y=packed["y"][selected] % 3, group=packed["group"][selected])
    cli.train(cfg, output / "prepared", output / "resnet18", "nyquist")
    report = cli.evaluate(output / "resnet18/model.pt", output / "prepared/test.npz", output / "evaluation")
    np.save(output / "oracle.npy", dc)
    with patch("sys.argv", ["diagnostics", "--checkpoint", str(output / "resnet18/model.pt"),
                            "--reference", str(output / "resnet18/model.pt"), "--data", str(output / "prepared/test.npz"),
                            "--scales", "16", "32", "--oracle-support", str(output / "oracle.npy"),
                            "--output", str(output / "oracle_diagnostics.json")]):
        diagnostics.main()
    diagnostic = json.loads((output / "oracle_diagnostics.json").read_text())
    assert all(row["oracle_support_consistency"] is not None for row in diagnostic["resolutions"])
    np.savez(output / "x_only.npz", x=np.load(output / "prepared/test.npz")["x"])
    cli.infer(output / "resnet18/model.pt", output / "x_only.npz", output / "inference.npz")
    np.testing.assert_allclose(np.load(output / "inference.npz")["logits"],
                               np.load(output / "evaluation/predictions.npz")["logits"], atol=1e-7, rtol=1e-6)
    cli.plot([output / "resnet18", output / "evaluation"], output / "figures")
    summary = dict(seed=17, backbone="resnet18", branches=records, evaluation=report,
                   cifar_format_fixture=True, oracle_training_isolation=True, oracle=diagnostic)
    (output / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
