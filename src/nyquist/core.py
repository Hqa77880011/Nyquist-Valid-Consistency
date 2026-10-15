import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = ["nyquist", "erm", "multiscale", "full_consistency", "fixed_lowpass", "alias_only"]


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(128, cfg["classes"])

    def forward(self, x):
        features = self.encoder(x.float())
        return self.head(features.mean((-2, -1))), features


def spectral_pair(model, x, scale, method):
    low = F.interpolate(x, (scale, scale), mode="bilinear", align_corners=False, antialias=True)
    high_logits, high_feature = model(x)
    low_logits, low_feature = model(low)
    low_shape = low_feature.shape[-2:]
    low_feature = F.interpolate(
        low_feature, high_feature.shape[-2:], mode="bilinear", align_corners=False
    )
    high_fft = torch.fft.fft2(high_feature, norm="ortho")
    low_fft = torch.fft.fft2(low_feature, norm="ortho")
    h, w = high_feature.shape[-2:]
    fy, fx = torch.fft.fftfreq(h, device=x.device) * h, torch.fft.fftfreq(w, device=x.device) * w
    cutoff_y, cutoff_x = low_shape[0] / 2, low_shape[1] / 2
    if method == "fixed_lowpass":
        cutoff_y, cutoff_x = h / 4, w / 4
    mask = ((fy.abs()[:, None] < cutoff_y) & (fx.abs()[None] < cutoff_x)).to(high_fft.dtype)
    difference = high_fft - low_fft
    full = difference.abs().square().mean((1, 2, 3))
    projected = (difference * mask).abs().square().mean((1, 2, 3))
    image_fft = torch.fft.fft2(x, norm="ortho").abs().square()
    ih, iw = x.shape[-2:]
    iy, ix = (
        torch.fft.fftfreq(ih, device=x.device) * ih,
        torch.fft.fftfreq(iw, device=x.device) * iw,
    )
    unsupported = (iy.abs()[:, None] >= scale / 2) | (ix.abs()[None] >= scale / 2)
    alias = (image_fft * unsupported).sum((1, 2, 3)) / image_fft.sum((1, 2, 3)).clamp_min(1e-8)
    weight = (1 - alias.detach()).clamp(0, 1)
    consistency = full if method in ("full_consistency", "alias_only") else projected
    if method in ("nyquist", "alias_only"):
        consistency = weight * consistency
    supported = projected * weight if method in ("nyquist", "alias_only") else projected
    return (
        high_logits,
        low_logits,
        consistency.mean(),
        full.mean(),
        projected.mean(),
        supported.mean(),
    )


def loss(model, batch, cfg):
    high, low, consistency, _, _, _ = spectral_pair(
        model, batch["x"].float(), cfg["low_resolution"], model.method
    )
    task = F.cross_entropy(high, batch["y"].long())
    if model.method != "erm":
        task = 0.5 * (task + F.cross_entropy(low, batch["y"].long()))
    if model.method in ("erm", "multiscale"):
        return task
    return task + cfg["consistency_weight"] * consistency


def predict(model, batch, cfg, calibration=None):
    logits, _ = model(batch["x"])
    low = F.interpolate(
        batch["x"].float(),
        (cfg["low_resolution"], cfg["low_resolution"]),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    low_logits, _ = model(low)
    return {"logits": logits, "low_logits": low_logits}


def metrics(prediction, data, cfg):
    report = classification_metrics(prediction["logits"], data["y"])
    report["low_resolution"] = classification_metrics(prediction["low_logits"], data["y"])
    report["resolution_accuracy_gap"] = report["accuracy"] - report["low_resolution"]["accuracy"]
    return report


def inference(model, batch, cfg):
    return model(batch["x"])
