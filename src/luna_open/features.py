"""Frozen pretrained image encoders; no random-backbone fallback.

Sapiens preprocessing follows LHM's inspected wrapper. DINO uses official HF
normalization (an explicit difference from the inspected LHM face wrapper).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def resize_square(image: Tensor, size: int) -> Tensor:
    height, width = image.shape[-2:]
    side = max(height, width)
    dh, dw = side - height, side - width
    image = F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=1)
    return F.interpolate(image, (size, size), mode="bicubic", align_corners=True, antialias=True)


class SapiensFeatures(nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        self.model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.4844, 0.4570, 0.4062])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.2295, 0.2236, 0.2256])[None, :, None, None])

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        image = (resize_square(image, 1024) - self.mean) / self.std
        (features,) = self.model(image)
        if features.shape[1] != 1536:
            raise ValueError(f"Expected Sapiens-1B channels=1536, got {features.shape}")
        return features.flatten(2).transpose(1, 2).contiguous()


class DinoFeatures(nn.Module):
    def __init__(self, directory: str | Path, kind: str):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        if kind not in {"face", "motion"}:
            raise ValueError(kind)
        self.model = AutoModel.from_pretrained(directory, local_files_only=True).eval()
        self.model.requires_grad_(False)
        # DINOv3 exposes only the fast processor in transformers 4.57.6.
        # We read its normalization constants; resizing is performed below.
        processor = AutoImageProcessor.from_pretrained(
            directory, local_files_only=True, use_fast=kind == "motion"
        )
        self.register_buffer("mean", torch.tensor(processor.image_mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(processor.image_std)[None, :, None, None])
        self.kind = kind
        self.size = 448 if kind == "face" else 512
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        image = (resize_square(image, self.size) - self.mean) / self.std
        output = self.model(pixel_values=image, output_hidden_states=self.kind == "face")
        patches = (self.size // self.patch_size) ** 2
        if self.kind == "face":
            # HF hidden_states includes embedding at index 0. LHM uses zero-based
            # transformer layers [4,11,17,23], so take [5,12,18,24]. Each gets LN.
            layers = [
                self.model.layernorm(output.hidden_states[i])[:, -patches:] for i in (5, 12, 18, 24)
            ]
            return torch.stack(layers, dim=1)
        return output.last_hidden_state[:, -patches:]
