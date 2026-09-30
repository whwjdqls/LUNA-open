"""Export NeuMan SMPL surfaces for the official SMPL-to-SMPL-X mesh fitter.

These are source meshes, not converted SMPL-X parameters. Vertices are in the
target camera frame in native SMPL meters, with x right, y down and z forward.
Use the accompanying crop intrinsics for eventual baseline render comparison.
"""

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml

from luna_open.data.neuman import NeuManDataset
from luna_open.geometry import project_points, transform_points
from luna_open.provenance import file_sha256, verify_sources
from luna_open.smpl import SMPLTeacher


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new or empty mesh-export directory")
    cfg = yaml.safe_load(args.config.read_text())
    verify_sources(Path(cfg["data_root"]), json.loads(Path(cfg["manifest"]).read_text()))
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).eval()
    # smplx.SMPL.forward has no pose-corrective switch. Match this dataset's
    # explicit convention in this private export instance; asset bytes stay fixed.
    if not teacher.pose_blend_shapes:
        teacher.body.posedirs.zero_()
    data = NeuManDataset(
        cfg["data_root"], cfg["manifest"], split=args.split, size=cfg["image_size"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    faces = teacher.body.faces_tensor.numpy().astype(np.int64)
    records = []
    for scene, info in data.metadata.items():
        names = sorted(set(info["references"] + info["splits"][args.split]))
        (args.output / scene).mkdir()
        for name in names:
            frame = data.load_frame(scene, name)
            pose, betas, camera = [frame[k][None] for k in ("pose", "betas", "body_to_camera")]
            vertices = teacher.body(
                betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:]
            ).vertices
            samples, _ = teacher.deformation(pose, betas)
            sample_error = float((teacher.interpolate(vertices) - samples).abs().max())
            if sample_error > 2e-5:
                raise ValueError(f"Export and teacher geometry disagree: {scene}/{name}")
            camera_vertices = transform_points(camera, vertices)[0]
            pixels, positive_depth = project_points(camera_vertices, frame["K"])
            if not torch.isfinite(pixels).all() or not positive_depth.all():
                raise ValueError(f"Invalid camera-space export: {scene}/{name}")
            relative = Path(scene) / f"{Path(name).stem}.obj"
            with (args.output / relative).open("w") as stream:
                stream.write("# NeuMan SMPL mesh for parameter transfer; camera-frame meters\n")
                np.savetxt(stream, camera_vertices.numpy(), fmt="v %.9g %.9g %.9g")
                np.savetxt(stream, faces + 1, fmt="f %d %d %d")
            records.append(
                dict(
                    scene=scene,
                    frame=name,
                    role="reference" if name in info["references"] else "target",
                    mesh=str(relative),
                    sha256=file_sha256(args.output / relative),
                    vertices=len(camera_vertices),
                    faces=len(faces),
                    teacher_surface_max_abs_error_m=sample_error,
                    min_depth_m=float(camera_vertices[:, 2].min()),
                    max_depth_m=float(camera_vertices[:, 2].max()),
                    K=frame["K"].tolist(),
                    width=cfg["image_size"],
                    height=cfg["image_size"],
                    original_alignment_scale=float(frame["alignment_scale"]),
                )
            )
        print(scene, len(names), "meshes exported", flush=True)
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        split=args.split,
        config=str(args.config),
        manifest_sha256=file_sha256(cfg["manifest"]),
        smpl_asset_sha256=file_sha256(cfg["smpl_model"]),
        smplx_library_version=version("smplx"),
        pose_blend_shapes=teacher.pose_blend_shapes,
        coordinate_frame="native SMPL meters per camera; x right, y down, z forward",
        references={scene: info["references"] for scene, info in data.metadata.items()},
        targets={scene: info["splits"][args.split] for scene, info in data.metadata.items()},
        meshes=records,
        limitation="Source geometry only; not fitted SMPL-X or native baseline results",
    )
    (args.output / "meshes.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Exported", len(records), "source meshes; SMPL-X fitting remains required", flush=True)


if __name__ == "__main__":
    main()
