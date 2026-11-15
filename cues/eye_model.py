"""MobileNetV2 eye-state classifier and its CPU predictor (R4, Milestone 4).

Model
-----
ImageNet-pretrained MobileNetV2 (transfer learning) with a new head:
global average pool -> dropout -> one linear unit = logit of P(eye closed).

* Input: (B, 1, 64, 64) float in [0, 1], CLAHE-normalised grayscale crops.
  Inside the model the crop is upsampled to ``input_size`` (default 96),
  copied to 3 channels and normalised with ImageNet mean/std, because the
  pretrained filters expect 3-channel ImageNet-like statistics.
* A learned ``temperature`` (temperature scaling, Guo et al. 2017) is fitted
  on the validation people after training, so the probabilities are
  calibrated: "P(closed) = 0.8" is right about 80% of the time. R7's
  reliability estimator and R6's GRU rely on that.
* ImageNet weights only exist for width_mult = 1.0. A 0.5-width network is
  supported but trains from scratch (``pretrained: false``).

Predictor
---------
``EyeStatePredictor(checkpoint).predict_crops(crops_uint8)`` -> P(closed)
for a batch of crops from ``eye_crop.crop_eye``. Left + right eye go in one
batch of 2 per frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_FORMAT = 1


@dataclass
class EyeModelConfig:
    input_size: int = 96
    width_mult: float = 1.0
    dropout: float = 0.2
    pretrained: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "EyeModelConfig":
        d = {k: v for k, v in dict(d or {}).items() if k in {f.name for f in fields(cls)}}
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EyeNet(nn.Module):
    def __init__(self, cfg: Optional[EyeModelConfig] = None):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        self.cfg = cfg or EyeModelConfig()
        if self.cfg.pretrained and self.cfg.width_mult != 1.0:
            raise ValueError("ImageNet weights exist only for width_mult=1.0; set pretrained: false")
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if self.cfg.pretrained else None
        backbone = mobilenet_v2(weights=weights, width_mult=self.cfg.width_mult)
        self.features = backbone.features
        self.head = nn.Sequential(nn.Dropout(self.cfg.dropout), nn.Linear(backbone.last_channel, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) in [0, 1] -> raw logits (B,), without temperature."""
        s = self.cfg.input_size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(f).squeeze(1)

    def calibrated_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) / self.temperature

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = trainable


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Single scalar T > 0 minimising validation log-loss of sigmoid(logit / T)."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    return t if np.isfinite(t) and 0.05 < t < 20 else 1.0


def save_checkpoint(path: str | Path, model: EyeNet, extra: Optional[dict] = None) -> None:
    """Checkpoint with only plain Python types + tensors (loadable with weights_only=True)."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "temperature": float(model.temperature.item()),
        "extra": _plain(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[EyeNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = EyeModelConfig.from_dict(ckpt["model_config"])
    cfg.pretrained = False          # weights come from the checkpoint, no download needed
    model = EyeNet(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def _plain(obj: Any) -> Any:
    """Convert numpy scalars / containers to plain Python for safe checkpoint loading."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class EyeStatePredictor:
    """CPU inference for the live pipeline (R8) and offline evaluation (R3).

    >>> pred = EyeStatePredictor("checkpoints/eye_cnn/best.pt")
    >>> p_closed = pred.predict_crops(np.stack([left_crop, right_crop]))   # (2,) float32
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", num_threads: Optional[int] = None):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.temperature = float(ckpt.get("temperature", 1.0))

    @torch.inference_mode()
    def predict_crops(self, crops: np.ndarray) -> np.ndarray:
        """crops: (B, 64, 64) uint8 CLAHE-normalised -> (B,) float32 calibrated P(closed)."""
        crops = np.asarray(crops)
        if crops.ndim == 2:
            crops = crops[None]
        if crops.ndim != 3:
            raise ValueError(f"expected (B, H, W) crops, got {crops.shape}")
        x = torch.from_numpy(np.ascontiguousarray(crops, dtype=np.uint8)).to(self.device)
        x = x.float().div_(255.0).unsqueeze(1)
        p = torch.sigmoid(self.model.calibrated_logits(x))
        return p.float().cpu().numpy()
