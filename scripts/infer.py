"""Render a trained image-driven avatar, without SMPL parameters or body assets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from luna_open.pipeline import LUNAPipeline


def load_image(path: Path, size: int):
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (max(image.size),) * 2, "white")
    canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
    array = np.asarray(canvas.resize((size, size), Image.Resampling.BICUBIC)).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs=4, required=True)
    parser.add_argument("--faces", type=Path, nargs=4)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument(
        "--camera", type=Path, required=True, help="JSON with K, width, height for the driving crop"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run inference in a Slurm GPU allocation")
    model = LUNAPipeline.from_checkpoint(args.checkpoint, args.assets)
    references = torch.stack([load_image(p, 1024) for p in args.references])[None].cuda()
    faces = (
        torch.stack([load_image(p, 448) for p in args.faces])[None].cuda() if args.faces else None
    )
    driver = load_image(args.driver, 512)[None].cuda()
    camera = json.loads(args.camera.read_text())
    K = torch.tensor(camera["K"], dtype=torch.float32, device="cuda")[None]
    avatar = model.encode_identity(references, faces)
    posed = model.animate(avatar, driver)
    result = model.render(posed, K, (camera["height"], camera["width"]))
    args.output.mkdir(parents=True, exist_ok=True)
    image = (result["rgb"][0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    alpha = (result["alpha"][0, 0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(image).save(args.output / "rgb.png")
    Image.fromarray(alpha).save(args.output / "alpha.png")
    np.savez_compressed(
        args.output / "gaussians.npz",
        **{
            name: getattr(posed.gaussians, name)[0].cpu().numpy()
            for name in ("means", "quaternions", "scales", "opacities", "colors")
        },
    )
    (args.output / "inference.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(args.checkpoint),
                references=list(map(str, args.references)),
                driver=str(args.driver),
                camera=camera,
                face_crops="provided" if args.faces else "upper35percent_fallback",
                gaussian_frame="driving camera; meters; wxyz",
                preprocessing="foreground crops expected; white square padding",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
