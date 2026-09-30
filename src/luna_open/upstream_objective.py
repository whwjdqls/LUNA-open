"""Execute released LHM++ training loss code for the shared comparison objective.

Attribution: Tongyi Lab / Alibaba, LHM-plusplus, Apache-2.0, commit
906b5d9fb967ab42efb92f6fa55bf22cac86b653. The source remains in an external
checkout. Select the unmodified loss methods with Python's AST so LUNA does not
need to import the upstream dataset, distributed runner or CUDA point modules.
This calls released training code; it does not reproduce its training schedule.
"""

import ast
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

COMMIT = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


def load_class(path: Path, name: str, methods=None):
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is not None:
        definition.bases = []
        definition.decorator_list = []
        definition.body = [
            node
            for node in definition.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        if {node.name for node in definition.body} != set(methods):
            raise ValueError("Released loss method is absent")
    # Upstream PixelLoss uses torch.compile; compilation is disabled for all
    # comparison methods. Preserve its implementation and avoid a global import.
    for node in ast.walk(definition):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [
                decorator
                for decorator in node.decorator_list
                if not (
                    isinstance(decorator, ast.Attribute)
                    and isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "torch"
                    and decorator.attr == "compile"
                )
            ]
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], hashlib.sha256(text.encode()).hexdigest()


class ReleasedPhotometricObjective:
    def __init__(self, source: Path, weights: dict, lpips):
        source = source.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != COMMIT:
            raise ValueError("LHM++ training source pin differs")
        subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
        trainer, trainer_sha = load_class(
            source / "core/runners/train/human_lrm_a4o.py",
            "HumanLRMA4OTrainer",
            {"forward_loss_local_step", "get_smplx_params", "get_loss_weight"},
        )
        pixel, pixel_sha = load_class(source / "core/losses/pixelwise.py", "PixelLoss")
        self.handle = trainer()
        self.handle.cfg = SimpleNamespace(
            train=SimpleNamespace(
                loss=AttributeDict(
                    pixel_weight=weights["rgb"],
                    mask_weight=weights["mask"],
                    perceptual_weight=weights["lpips"],
                    face_id_weight=0.0,
                )
            )
        )
        self.handle.pixel_loss_fn = pixel("l1")
        self.handle.ball_loss = self.handle.offset_loss = self.handle.opacity_loss = None
        self.lpips = lpips
        self.handle.perceptual_loss_fn = self.perceptual
        self.provenance = dict(
            commit=revision,
            trainer_sha256=trainer_sha,
            pixel_sha256=pixel_sha,
            methods=["forward_loss_local_step", "get_smplx_params", "get_loss_weight"],
            compilation="disabled",
            regularizers="disabled equally in shared_photometric",
        )

    def perceptual(self, prediction, target):
        prediction = prediction.flatten(0, 1)
        target = target.flatten(0, 1)
        return self.lpips(2 * prediction - 1, 2 * target - 1).mean()

    def __call__(self, prediction: dict, rgb, mask):
        if prediction["rgb"].shape != rgb.shape or prediction["alpha"].shape != mask.shape:
            raise ValueError("Prediction and shared target canvas differ")
        # The released trainer supervises comp_rgb / comp_mask. LHM++ uses its
        # final DPT predictions here, ensuring that the renderer gets gradients.
        outputs = dict(
            comp_rgb=prediction["rgb"][:, None],
            comp_mask=prediction["alpha"][:, None],
            mesh_meta=None,
        )
        self.handle.model = lambda **kwargs: outputs
        data = dict(
            render_image=rgb[:, None],
            render_mask=mask[:, None],
            render_padding_mask=torch.zeros_like(mask[:, None]),
            render_head_mask=torch.zeros_like(mask[:, None]),
            use_heads=None,
            source_rgbs=None,
            source_head_rgbs=None,
            c2ws=None,
            intrs=None,
            render_bg_colors=None,
            real_num_view=1,
        )
        result = self.handle.forward_loss_local_step(data)
        self.handle.model = None  # release the graph between microsteps
        return result["loss"], {
            "rgb": result["loss_pixel"],
            "mask": result["loss_mask"],
            "lpips": result["loss_perceptual"],
        }
