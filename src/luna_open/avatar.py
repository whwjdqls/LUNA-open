from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class Gaussians:
    """Batched tensors [B,K,D]; opacity [B,K]; canonical or camera-space meters."""

    means: Tensor
    quaternions: Tensor
    scales: Tensor
    opacities: Tensor
    colors: Tensor

    def detach(self) -> "Gaussians":
        return Gaussians(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def validate(self) -> None:
        b, n, d = self.means.shape
        assert d == 3
        for name, shape in (
            ("quaternions", (b, n, 4)),
            ("scales", (b, n, 3)),
            ("colors", (b, n, 3)),
            ("opacities", (b, n)),
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {getattr(self, name).shape}")
        if not all(torch.isfinite(getattr(self, f.name)).all() for f in fields(self)):
            raise ValueError("Nonfinite Gaussian attribute")
        if (self.scales <= 0).any():
            raise ValueError("Gaussian scales must be positive")


@dataclass
class CanonicalAvatar:
    gaussians: Gaussians
    tokens: Tensor


@dataclass
class PosedAvatar:
    gaussians: Gaussians
    global_quaternion: Tensor
    translation: Tensor
    rotation_sincos: Tensor
