"""Verify the licensed SMPL teacher and NeuMan's distinct pose convention.

The upstream implementation is imported read-only from the pinned checkout.
Projection/mask diagnostics describe supplied fits, not reconstruction scores.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.geometry import project_points, transform_points
from luna_open.provenance import file_sha256, verify_sources
from luna_open.smpl import SMPLTeacher


def overlay(sample, pixels, title):
    picture = Image.fromarray((sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    for x, y in pixels[::4].tolist():
        if 0 <= x < 512 and 0 <= y < 512:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 150, 255))
    draw.rectangle((0, 0, 512, 24), fill="white")
    draw.text((8, 6), title, fill="black")
    return picture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smpl", type=Path, required=True)
    parser.add_argument("--neuman-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_sources(args.root, json.loads(args.manifest.read_text()))
    source = args.neuman_code / "models/smpl.py"
    spec = importlib.util.spec_from_file_location("neuman_reference_smpl", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = module.SMPL(str(args.smpl), device=torch.device("cpu")).eval()
    teacher = SMPLTeacher(str(args.smpl), pose_blend_shapes=False).eval()
    standard = SMPLTeacher(str(args.smpl), pose_blend_shapes=True).eval()
    data = NeuManDataset(args.root, args.manifest)
    summaries, records, thumbnails = {}, [], []
    for scene, info in data.metadata.items():
        scene_rows, corrective_distances = [], []
        for index, row in enumerate(info["frames"]):
            sample = data.load_frame(scene, row["name"])
            pose, beta, camera = [sample[k][None] for k in ("pose", "betas", "body_to_camera")]
            with torch.no_grad():
                library = standard.body(
                    betas=beta, global_orient=pose[:, :3], body_pose=pose[:, 3:]
                ).vertices
                standard_surface, _ = standard.deformation(pose, beta)
                expected = upstream(poses=pose, betas=beta)[None]
                surface, _ = teacher.deformation(pose, beta)
                standard_error = float(
                    (standard_surface - teacher.interpolate(library)).abs().max()
                )
                neuman_error = float((surface - teacher.interpolate(expected)).abs().max())
                if max(standard_error, neuman_error) > 2e-5:
                    raise ValueError(f"SMPL deformation disagrees: {scene}/{row['name']}")
                corrective_distances.extend((library - expected).norm(dim=-1).flatten().tolist())
                camera_vertices = transform_points(camera, expected)[0]
                pixel, valid = project_points(camera_vertices, sample["K"])
                standard_pixel, _ = project_points(
                    transform_points(camera, library)[0], sample["K"]
                )
                if not torch.isfinite(pixel).all() or not valid.all():
                    raise ValueError(f"Invalid posed mesh projection: {scene}/{row['name']}")
                xy = pixel.round().long()
                inside = ((xy >= 0) & (xy < 512)).all(-1)
                covered = torch.zeros(len(xy), dtype=torch.bool)
                covered[inside] = sample["mask"][0, xy[inside, 1], xy[inside, 0]] > 0.5
                record = dict(
                    scene=scene,
                    frame=row["name"],
                    standard_surface_max_abs_error_m=standard_error,
                    neuman_surface_max_abs_error_m=neuman_error,
                    minimum_depth_m=float(camera_vertices[:, 2].min()),
                    projected_vertex_foreground_fraction=float(covered.float().mean()),
                    pose_corrective_projection_mean_px=float(
                        (pixel - standard_pixel).norm(dim=-1).mean()
                    ),
                )
                if index == 0:
                    # Verify the exact DA-pose detour used by NeuManReader.
                    da_pose = torch.zeros_like(pose)
                    da_pose[:, 5], da_pose[:, 8] = 1, -1
                    _, target_t = upstream.verts_transformations(poses=pose, betas=beta)
                    _, da_t = upstream.verts_transformations(poses=da_pose, betas=beta)
                    da_vertices = upstream(poses=da_pose, betas=beta)[None]
                    composed = target_t @ torch.linalg.inv(da_t)
                    via_da = (composed[..., :3, :3] @ da_vertices[..., None]).squeeze(
                        -1
                    ) + composed[..., :3, 3]
                    da_error = float((via_da - expected).abs().max())
                    if da_error > 2e-5:
                        raise ValueError("NeuMan DA-pose roundtrip failed")
                    record["neuman_da_roundtrip_max_abs_error_m"] = da_error
                    picture = overlay(sample, pixel, f"{scene}: optimized fit, NeuMan convention")
                    picture.save(args.output / f"{scene}-optimized-overlay.png")
                    thumbnails.append(picture)
                scene_rows.append(record)
                records.append(record)
        summaries[scene] = dict(
            frames=len(scene_rows),
            standard_surface_max_abs_error_m=max(
                r["standard_surface_max_abs_error_m"] for r in scene_rows
            ),
            neuman_surface_max_abs_error_m=max(
                r["neuman_surface_max_abs_error_m"] for r in scene_rows
            ),
            pose_corrective_displacement_mean_m=float(np.mean(corrective_distances)),
            pose_corrective_displacement_p95_m=float(np.percentile(corrective_distances, 95)),
            projected_vertex_foreground_fraction_mean=float(
                np.mean([r["projected_vertex_foreground_fraction"] for r in scene_rows])
            ),
            pose_corrective_projection_mean_px=float(
                np.mean([r["pose_corrective_projection_mean_px"] for r in scene_rows])
            ),
        )
        print(json.dumps({scene: summaries[scene]}), flush=True)

    # Real-body teacher must train the canonical means while detaching targets
    # for animator distillation; rotations must remain proper quaternions.
    means = teacher.shaped_anchors(beta).requires_grad_()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, len(teacher.anchors), 4)
    gaussians = Gaussians(
        means,
        q,
        torch.ones_like(means) * 0.008,
        torch.ones_like(means[..., 0]),
        torch.ones_like(means) * 0.5,
    )
    posed = teacher(gaussians, pose, beta, camera, detach=False)
    posed.means.square().mean().backward()
    if means.grad is None or not torch.isfinite(means.grad).all() or means.grad.abs().sum() <= 0:
        raise ValueError("Real teacher has no finite canonical-mean gradient")
    if teacher(gaussians, pose, beta, camera).means.requires_grad:
        raise ValueError("Teacher target is not detached")
    torch.testing.assert_close(
        posed.quaternions.norm(dim=-1), torch.ones_like(posed.opacities), atol=1e-5, rtol=1e-5
    )
    montage = Image.new("RGB", (1536, 1024), "white")
    for i, picture in enumerate(thumbnails):
        montage.paste(picture, ((i % 3) * 512, (i // 3) * 512))
    montage.save(args.output / "optimized-overlays.png")
    result = dict(
        smpl_asset_sha256=file_sha256(args.smpl),
        manifest_sha256=file_sha256(args.manifest),
        upstream_smpl_sha256=file_sha256(source),
        num_queries=len(teacher.anchors),
        pose_blend_shapes=False,
        canonical_gradient_l1=float(means.grad.abs().sum()),
        teacher_detach_verified=True,
        seconds=time.perf_counter() - started,
        scenes=summaries,
        frames=records,
        limitation="Supplied-fit diagnostic; foreground vertex fraction is not silhouette IoU or a learned-model score",
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"scenes", "frames"}}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
