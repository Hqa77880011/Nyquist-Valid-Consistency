import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = ["nyquist", "erm", "multiscale", "full_consistency", "fixed_lowpass", "alias_only", "blurpool"]


class BlurPool(nn.Module):
    """Fixed binomial low-pass before subsampling; no learned parameters."""

    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1., 2., 1.])
        kernel = kernel[:, None] * kernel[None, :] / 16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernel, stride=2, groups=x.shape[1])


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        backbone = cfg.get("backbone", "cnn")
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
        if method == "blurpool":
            self.encoder[3].stride = (1, 1)
            self.encoder.insert(6, BlurPool(128))
        if backbone == "resnet18":
            from torchvision.models import resnet18

            network = resnet18(weights=None)
            network.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.encoder = nn.Sequential(
                network.conv1, network.bn1, network.relu,
                network.layer1, network.layer2, network.layer3,
            )
            self.head = nn.Linear(256, cfg["classes"])
            if method == "blurpool":
                for parent in list(self.encoder.modules()):
                    for name, layer in list(parent.named_children()):
                        if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                            layer.stride = (1, 1)
                            parent._modules[name] = nn.Sequential(layer, BlurPool(layer.out_channels))
        elif backbone != "cnn":
            raise ValueError("backbone must be cnn or resnet18")

    def forward(self, x):
        features = self.encoder(x.float())
        return self.head(features.mean((-2, -1))), features


def spectral_pair(model, x, scale, method, task_support=None):
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
    common = ((fy.abs()[:, None] < cutoff_y) & (fx.abs()[None] < cutoff_x)).to(high_fft.dtype)
    mask = common
    if method == "fixed_lowpass":
        mask = ((fy.abs()[:, None] < h / 4) & (fx.abs()[None] < w / 4)).to(high_fft.dtype)
    if method == "oracle_support":
        if task_support is None:
            raise ValueError("The evaluation-only oracle needs externally supplied task support")
        support = torch.as_tensor(task_support, device=x.device)
        if support.ndim == 2:
            support = support[None].expand(len(x), -1, -1)
        if support.ndim != 3 or len(support) != len(x) or not torch.all((support == 0) | (support == 1)):
            raise ValueError("Task support must be binary (H,W) or (N,H,W), in unshifted FFT order")
        # Frequencies are integer cycles on the common physical domain, not resize indices.
        rows = fy.long().remainder(support.shape[-2])
        columns = fx.long().remainder(support.shape[-1])
        mask = support[:, rows[:, None], columns[None, :]][:, None].to(high_fft.dtype)
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
    consistency = full if method in ("full_consistency", "alias_only", "blurpool") else projected
    if method in ("nyquist", "alias_only", "oracle_support"):
        consistency = weight * consistency
    applied_mask = torch.ones_like(common) if method in ("full_consistency", "alias_only", "blurpool") else mask
    supported = (difference * applied_mask * common).abs().square().mean((1, 2, 3))
    if method in ("nyquist", "alias_only", "oracle_support"):
        supported = supported * weight
    return (
        high_logits,
        low_logits,
        consistency.mean(),
        full.mean(),
        projected.mean(),
        supported.mean(),
    )


def loss(model, batch, cfg):
    if model.method == "erm":
        return F.cross_entropy(model(batch["x"])[0], batch["y"].long())
    if model.method == "multiscale":
        low = F.interpolate(batch["x"].float(), (cfg["low_resolution"], cfg["low_resolution"]),
                            mode="bilinear", align_corners=False, antialias=True)
        return 0.5 * (F.cross_entropy(model(batch["x"])[0], batch["y"].long()) +
                      F.cross_entropy(model(low)[0], batch["y"].long()))
    high, low, consistency, _, _, _ = spectral_pair(
        model, batch["x"].float(), cfg["low_resolution"],
        "oracle_support" if model.method == "oracle_reference" else model.method,
        batch.get("oracle_support") if model.method == "oracle_reference" else None,
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
