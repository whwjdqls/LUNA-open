"""Gradient-enabled adapters calling the pinned native LHM / LHM++ models.

LHM: Alibaba Apache-2.0, 4f88aaeb3629249fbbddb4d0784a06962d9e1338.
LHM++: Alibaba Apache-2.0, 906b5d9fb967ab42efb92f6fa55bf22cac86b653.
Body templates, point queries, transformer layers, frozen encoders and LHM++
DPT remain native. The adapter supplies explicit image dimensions because the
released top-level forward derives dimensions from the camera principal point;
that assumption is invalid for the common off-center crops.
"""

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .provenance import file_sha256

PINS = dict(
    lhm="4f88aaeb3629249fbbddb4d0784a06962d9e1338", lhmpp="906b5d9fb967ab42efb92f6fa55bf22cac86b653"
)


def blank_motion(device, batch=1):
    shapes = dict(
        root_pose=(3,),
        body_pose=(21, 3),
        jaw_pose=(3,),
        leye_pose=(3,),
        reye_pose=(3,),
        lhand_pose=(15, 3),
        rhand_pose=(15, 3),
        expr=(100,),
        trans=(3,),
    )
    result = {name: torch.zeros(batch, 1, *shape, device=device) for name, shape in shapes.items()}
    result["betas"] = torch.zeros(batch, 10, device=device)
    return result


def remove_unused_dpt_blocks(model, config):
    """Only the inherited blocks absent from and unused by the DPT-only release."""
    if config["neural_renderer"]["type"] != "patch_4dptonly":
        raise ValueError("The selected LHM++ adapter requires the released DPT-only model")
    if model.neural_renderer.__class__.__name__ != "PatchDPT4DecoderOnly":
        raise ValueError("Unexpected DPT implementation")
    keys = list(model.neural_renderer.transformer_block.state_dict())
    model.neural_renderer.transformer_block = nn.ModuleList()
    return keys


class NativeReconstruction(nn.Module):
    def __init__(self, method, source: Path, runtime: Path, architecture: Path):
        super().__init__()
        if method not in PINS:
            raise ValueError("Unknown native baseline")
        source, runtime, architecture = source.resolve(), runtime.resolve(), architecture.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != PINS[method]:
            raise ValueError("Native model source pin differs")
        subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
        sys.path.insert(0, str(source))
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        torch._dynamo.config.disable = True
        from accelerate import PartialState

        PartialState()
        config = json.loads(architecture.read_text())
        config["use_face_id"] = False  # shared objective has no face-ID term
        config["encoder_freeze"] = True  # same pretrained-backbone policy as LUNA
        if method == "lhm":
            config["fine_encoder_freeze"] = True
        if method == "lhmpp":
            # The inference config freezes these freshly initialized internal
            # reconstruction networks. They must train in a fresh comparison.
            config["transformer_decoder"]["freeze_point"] = False
            config["transformer_decoder"]["freeze_image"] = False
        cwd = Path.cwd()
        try:
            os.chdir(runtime)
            if method == "lhm":
                import torchvision.transforms.functional as functional

                sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)
                from LHM.models.modeling_human_lrm import ModelHumanLRMSapdinoBodyHeadSD3_5
                from LHM.models.rendering.gsplat_renderer import GSPlatRenderer

                self.model = ModelHumanLRMSapdinoBodyHeadSD3_5(**config)
                self.model.renderer.get_gaussians_properties = types.MethodType(
                    GSPlatRenderer.get_gaussians_properties, self.model.renderer
                )
                self.model.renderer.forward_single_view = types.MethodType(
                    GSPlatRenderer.forward_single_view, self.model.renderer
                )
                removed = []
            else:
                from core.models.modeling_humana4o_lrm import ModelHumanA4OLRM

                self.model = ModelHumanA4OLRM(**config)
                removed = remove_unused_dpt_blocks(self.model, config)
        finally:
            os.chdir(cwd)
        self.method = method
        receipt_path = runtime / "runtime.json"
        asset_hashes = {}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            for name, saved in receipt["files"].items():
                current = file_sha256(runtime / name)
                if current != saved["sha256"]:
                    raise ValueError(f"Native runtime asset changed: {name}")
                asset_hashes[name] = current
        else:
            # Yonsei's source-copy layout has directory links and no runtime.json.
            for asset in sorted((runtime / "pretrained_models").rglob("*")):
                if asset.is_file():
                    asset_hashes[str(asset.relative_to(runtime))] = file_sha256(asset)
        dino = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vitl14_reg4_pretrain.pth"
        asset_hashes["dinov2_vitl14_reg4_pretrain.pth"] = file_sha256(dino)
        self.provenance = dict(
            method=method,
            source_commit=revision,
            config=config,
            initialization="fresh_reconstruction; native pretrained frozen backbones",
            removed_unused_dpt_constructor_keys=removed,
            runtime_asset_sha256=asset_hashes,
            training_policy="train all fresh reconstruction modules; freeze native pretrained image backbones",
            rendering="native renderer; explicit canvas size and unchanged crop K",
        )

    def train(self, mode=True):
        super().train(mode)
        # Pretrained frozen networks retain their statistics throughout training.
        for name in ("encoder", "fine_encoder", "faceESRGAN", "id_face_net"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                for child in module.modules():
                    if not any(p.requires_grad for p in child.parameters()):
                        child.eval()
        return self

    def forward(self, references, faces, motion, world_to_camera, K, size):
        """Batch one; metric body/world frame; poses [1,1,...], betas [1,10]."""
        if references.shape[0] != 1 or not torch.is_grad_enabled() and self.training:
            raise ValueError("Training expects batch-one microsteps with enabled gradients")
        if self.method == "lhm" and references.shape[1] != 1:
            raise ValueError("Native LHM requires exactly one reference")
        initial = blank_motion(references.device)
        motion = dict(motion)
        height, width = size
        if self.method == "lhm":
            if faces is None or faces.shape[:2] != references.shape[:2]:
                raise ValueError("LHM requires matching reference face crops")
            face = self.model.obtain_facesr(faces) if self.model.facesr else faces
            query, initial = self.model.renderer.get_query_points(initial, references.device)
            latent, images = self.model.forward_latent_points(
                references[:, 0], face[:, 0], None, query
            )
            attributes, query, initial = self.model.renderer.forward_gs(
                latent,
                query,
                initial,
                additional_features={"image_feats": images, "image": references[:, 0]},
            )
            padded_h, padded_w = height, width
            extra = {}
        else:
            query, initial = self.model.renderer.get_query_points(
                None, initial, device=references.device
            )
            latent, images, embedding, positions, _, _ = self.model.forward_latent_points(
                references,
                None,
                query,
                ref_imgs_bool=torch.ones(
                    references.shape[:2], dtype=torch.bool, device=references.device
                ),
            )
            attributes, query, initial = self.model.renderer.forward_gs(
                latent,
                query,
                initial,
                additional_features={"image_feats": images, "image": references[:, 0]},
            )
            patch = self.model.neural_renderer.enc_patch_size
            padded_h, padded_w = math.ceil(height / patch) * patch, math.ceil(width / patch) * patch
            extra = dict(features=latent, patch_size=self.model.neural_renderer_patch_size)
        motion["transform_mat_neutral_pose"] = initial["transform_mat_neutral_pose"]
        rendered = self.model.renderer.forward_animate_gs(
            attributes,
            query,
            motion,
            torch.linalg.inv(world_to_camera)[:, None],
            K[:, None],
            padded_h,
            padded_w,
            torch.ones(1, 1, 3, device=references.device),
            **extra,
        )
        if self.method == "lhmpp":
            rgb, alpha = self.model.neural_renderer(
                images,
                rendered["comp_features"],
                embedding,
                padded_h,
                padded_w,
                pos_emb_list=positions,
            )
            rgb = rgb[:, 0].permute(0, 3, 1, 2)
            alpha = alpha[:, 0]
            if alpha.ndim == 4:
                alpha = alpha.permute(0, 3, 1, 2)
            else:
                alpha = alpha[:, None]
        else:
            rgb, alpha = rendered["comp_rgb"][:, 0], rendered["comp_mask"][:, 0]
        return dict(rgb=rgb[..., :height, :width], alpha=alpha[..., :height, :width])
