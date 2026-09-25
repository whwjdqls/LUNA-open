"""Audit real cameras/alignment using ROMP meshes already in the NeuMan archive.

This does not validate optimized-SMPL forward deformation without the body asset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from luna_open.data.neuman import NeuManDataset, frame_annotation
from luna_open.geometry import project_points, transform_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = NeuManDataset(args.root, args.manifest, size=512)
    report, thumbnails = {}, []
    for scene, info in data.metadata.items():
        errors, crop_roundtrips, pose_disagreement = [], [], []
        for i, row in enumerate(info["frames"]):
            name = row["name"]
            archive = args.root / scene / "smpl_pred" / (Path(name).stem + "_png.npz")
            # Verified official-archive data; object deserialization is intentional.
            saved = np.load(archive, allow_pickle=True)["results"][0]
            annotation = frame_annotation(args.root / scene, row)
            transform = torch.from_numpy(annotation["body_to_camera"])
            joints = transform_points(transform, torch.from_numpy(saved["j3d_all54"]).float())
            full, valid = project_points(joints, torch.tensor(row["K"], dtype=torch.float32))
            error = np.linalg.norm(full[valid].numpy() - saved["pj2d_org"][valid.numpy()], axis=-1)
            errors.extend(error.tolist())
            sample = data.load_frame(scene, name)
            projected, _ = project_points(joints, sample["K"])
            mapped = torch.cat((full, torch.ones(len(full), 1)), -1) @ sample["crop_transform"].T
            crop_roundtrips.append(float((projected - mapped[:, :2]).abs().max()))
            saved_pose = saved["poses"] if "poses" in saved else saved["pose"]
            pose_disagreement.append(float(np.linalg.norm(annotation["pose"] - saved_pose)))
            if i == 0:
                rgb = (sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                overlay = Image.fromarray(rgb)
                drawing = ImageDraw.Draw(overlay)
                mesh = transform_points(transform, torch.from_numpy(saved["verts"]).float())
                pixel, in_front = project_points(mesh, sample["K"])
                for x, y in pixel[in_front][::4].tolist():
                    if 0 <= x < 512 and 0 <= y < 512:
                        drawing.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 150, 255))
                drawing.rectangle((0, 0, 512, 24), fill="white")
                drawing.text(
                    (8, 6), f"{scene}: supplied ROMP projection (not optimized fit)", fill="black"
                )
                overlay.save(args.output / f"{scene}-romp-overlay.png")
                thumbnails.append(overlay)
        report[scene] = dict(
            frames=len(info["frames"]),
            source_joint_projection_median_px=float(np.median(errors)),
            source_joint_projection_p95_px=float(np.percentile(errors, 95)),
            crop_projection_roundtrip_max_px=max(crop_roundtrips),
            optimized_vs_romp_pose_mean_l2=float(np.mean(pose_disagreement)),
        )
        if max(crop_roundtrips) > 0.01:
            raise ValueError(f"Crop projection mismatch: {scene}")
    montage = Image.new("RGB", (3 * 512, 2 * 512), "white")
    for i, picture in enumerate(thumbnails):
        montage.paste(picture, ((i % 3) * 512, (i // 3) * 512))
    montage.save(args.output / "romp-overlays.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
