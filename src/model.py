"""
NiSNN-A model and ablation variants for experiments c1–c4.

All three models share:
  - Input shape: (B, C=20, S=20, T=20)
  - Same head: Flatten → Linear(2000, 20) → ReLU → Linear(20, 2)
  - 2000 = 20 channels × 10 × 10 spatial dims (after MaxPool2d halves S and T)

BatchNorm2d is inserted before each NiLIF to prevent membrane-potential saturation.
Without it, leaky accumulation drives Ein far above the surrogate-gradient window
[vth, vth+1] and gradients go to zero, so training stalls near chance.

Variants:
  full          — Full NiSNN-A: learned encoder + learned classifier (c1/c2)
  encoder_only  — EncoderOnly:  learned encoder, no classifier block   (c3)
  fixed_encoder — FixedEncoder: fixed sign encoder, learned classifier  (c4)
"""

import torch
import torch.nn as nn
from torch import Tensor

from nilif import NiLIF


def _head(in_features: int = 2000) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


class NiSNNA(nn.Module):
    """Full NiSNN-A model (c1 / c2)."""

    def __init__(self):
        super().__init__()
        # Encoder: temporal filter within timepieces (kernel spans 5 of 20 T-steps)
        self.enc_conv  = nn.Conv2d(20, 20, kernel_size=(1, 5), padding=(0, 2))
        self.enc_bn    = nn.BatchNorm2d(20)
        self.enc_nilif = NiLIF(T=20)
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        # Classifier: global spatial-temporal integration over the (10,10) feature map
        self.clf_conv  = nn.Conv2d(20, 20, kernel_size=(10, 10), padding="same")
        self.clf_bn    = nn.BatchNorm2d(20)
        self.clf_nilif = NiLIF(T=10)  # T halved to 10 after MaxPool
        self.head      = _head()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 20, 20, 20)
        z = self.enc_conv(x)    # (B, 20, 20, 20)
        z = self.enc_bn(z)      # normalise → keeps Ein in surrogate-gradient window
        z = self.enc_nilif(z)   # (B, 20, 20, 20)
        z = self.pool(z)        # (B, 20, 10, 10)
        z = self.clf_conv(z)    # (B, 20, 10, 10)
        z = self.clf_bn(z)
        z = self.clf_nilif(z)   # (B, 20, 10, 10)
        return self.head(z)     # (B, 2)


class EncoderOnly(nn.Module):
    """Encoder-only ablation (c3): classifier block removed entirely."""

    def __init__(self):
        super().__init__()
        self.enc_conv  = nn.Conv2d(20, 20, kernel_size=(1, 5), padding=(0, 2))
        self.enc_bn    = nn.BatchNorm2d(20)
        self.enc_nilif = NiLIF(T=20)
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head      = _head()  # same 2000-dim input as full model

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc_conv(x)    # (B, 20, 20, 20)
        z = self.enc_bn(z)
        z = self.enc_nilif(z)   # (B, 20, 20, 20)
        z = self.pool(z)        # (B, 20, 10, 10)
        return self.head(z)     # (B, 2)


class FixedEncoder(nn.Module):
    """Fixed-encoder ablation (c4): sign threshold replaces the learned encoder."""

    def __init__(self):
        super().__init__()
        self.pool      = nn.MaxPool2d(kernel_size=2, stride=2)
        self.clf_conv  = nn.Conv2d(20, 20, kernel_size=(10, 10), padding="same")
        self.clf_bn    = nn.BatchNorm2d(20)
        self.clf_nilif = NiLIF(T=10)
        self.head      = _head()

    def forward(self, x: Tensor) -> Tensor:
        # Fixed sign threshold: fire wherever sample is positive.
        # After per-trial z-score this produces ~50% spikes with no class structure.
        z = (x > 0).float()    # (B, 20, 20, 20) — no gradient, no learned params
        z = self.pool(z)       # (B, 20, 10, 10)
        z = self.clf_conv(z)   # (B, 20, 10, 10)
        z = self.clf_bn(z)
        z = self.clf_nilif(z)  # (B, 20, 10, 10)
        return self.head(z)    # (B, 2)


def build_model(variant: str) -> nn.Module:
    variants = {
        "full":          NiSNNA,
        "encoder_only":  EncoderOnly,
        "fixed_encoder": FixedEncoder,
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(variants)}")
    return variants[variant]()
