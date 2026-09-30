"""Image-only LUNA inference. No body fitting or SMPL asset is used here.

Inputs are RGB foreground crops on white, with optional face crops. A trained
project checkpoint and actual pretrained encoder assets are required.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from .avatar import CanonicalAvatar, PosedAvatar
from .features import DinoFeatures, SapiensFeatures
from .model import IdentityEncoder, ModelConfig, NeuralAnimator
from .provenance import validate_inference_assets
from .rendering import render


class LUNAPipeline(nn.Module):
    def __init__(
        self,
        identity: IdentityEncoder,
        animator: NeuralAnimator,
        body_encoder: nn.Module,
        face_encoder: nn.Module,
        motion_encoder: nn.Module,
    ):
        super().__init__()
        self.identity = identity
        self.animator = animator
        self.body_encoder = body_encoder
        self.face_encoder = face_encoder
        self.motion_encoder = motion_encoder

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, assets: str | Path, device="cuda"):
        # Only load our own trusted training checkpoints: they contain optimizer
        # and RNG state and therefore cannot use weights_only=True.
        state = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if state["stage"] != "animator":
            raise ValueError("Image-driven inference needs an animator-stage checkpoint")
        assets = Path(assets)
        validate_inference_assets(state["feature_metadata"], assets)
        cfg = ModelConfig(**state["config"]["model"])
        identity = IdentityEncoder(
            state["identity"]["anchors"], state["identity"]["semantic_labels"], cfg
        )
        identity.load_state_dict(state["identity"])
        animator = NeuralAnimator(
            len(identity.anchors),
            cfg,
            state["animator"]["translation_mean"],
            state["animator"]["translation_std"],
        )
        animator.load_state_dict(state["animator"])
        pipeline = cls(
            identity,
            animator,
            SapiensFeatures(assets / "sapiens_body/sapiens_1b_epoch_173_torchscript.pt2"),
            DinoFeatures(assets / "dino_face", "face"),
            DinoFeatures(assets / "dino_motion", "motion"),
        )
        return pipeline.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_identity(
        self, references: Tensor, face_crops: Tensor | None = None
    ) -> CanonicalAvatar:
        """references [1,4,3,H,W]; optional face_crops [1,4,3,h,w]."""
        if references.ndim != 5 or references.shape[:3] != (1, 4, 3):
            raise ValueError("Inference currently expects one subject with four RGB references")
        if face_crops is None:
            # Explicit development fallback, not a learned face detector.
            height, width = references.shape[-2:]
            side = max(1, int(min(height, width) * 0.35))
            left = (width - side) // 2
            face_crops = references[..., :side, left : left + side]
        body, face = [], []
        with torch.autocast(references.device.type, dtype=torch.bfloat16):
            for view in range(4):
                body.append(self.body_encoder(references[:, view]))
                face.append(self.face_encoder(face_crops[:, view]))
            return self.identity(torch.stack(body, 1), torch.stack(face, 1))

    @torch.inference_mode()
    def animate(self, canonical: CanonicalAvatar, driving_image: Tensor) -> PosedAvatar:
        with torch.autocast(driving_image.device.type, dtype=torch.bfloat16):
            return self.animator(canonical, self.motion_encoder(driving_image))

    @torch.inference_mode()
    def render(
        self, posed: PosedAvatar, K: Tensor, size: tuple[int, int], viewmats: Tensor | None = None
    ):
        """K and optional viewmats render relative to the driving camera frame."""
        return render(posed.gaussians, K, size, viewmats)
