"""Measure supplied ROMP mesh motion in a fixed world frame; not model scores.

Uses all chronological frames, including training frames. This is an annotation
diagnostic, separate from the held-out reconstruction benchmark. FPS is unknown.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from luna_open.data.neuman import frame_annotation
from luna_open.geometry import transform_points
from luna_open.provenance import file_sha256, verify_sources
from luna_open.temporal import neuman_fixed_world, temporal_metrics, uniform_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_sources(args.root, manifest)
    source_hashes, scenes = {}, {}
    for scene, info in manifest["scenes"].items():
        camera_points, camera_matrices, scales, train_scales, ids = [], [], [], [], []
        for row in info["frames"]:
            path = args.root / scene / "smpl_pred" / (Path(row["name"]).stem + "_png.npz")
            source_hashes[str(path.relative_to(args.root))] = file_sha256(path)
            # Object arrays from the verified official archive; not arbitrary files.
            with np.load(path, allow_pickle=True) as archive:
                vertices = torch.from_numpy(archive["results"][0]["verts"]).double()
            if vertices.shape != (6890, 3):
                raise ValueError(f"SMPL vertex correspondence changed: {scene}/{row['name']}")
            annotation = frame_annotation(args.root / scene, row)
            transform = torch.from_numpy(annotation["body_to_camera"]).double()
            camera_points.append(transform_points(transform, vertices))
            camera_matrices.append(row["world_to_camera"])
            scale = float(annotation["alignment_scale"])
            scales.append(scale)
            ids.append(row["frame_id"])
            if row["name"] in info["splits"]["train"]:
                train_scales.append(scale)
        reference_scale = float(np.median(train_scales))
        interval = uniform_interval(torch.tensor(ids))
        positions = neuman_fixed_world(
            torch.stack(camera_points),
            torch.tensor(camera_matrices, dtype=torch.float64),
            torch.tensor(scales, dtype=torch.float64),
            reference_scale,
        )
        scenes[scene] = dict(
            **temporal_metrics(positions, interval),
            frames=len(ids),
            points=6890,
            frame_index_interval=interval,
            reference_scale=reference_scale,
            reference_scale_source="median of training-frame alignment scales",
        )
    report = dict(
        diagnostic="supplied ROMP vertex motion; NOT LUNA or baseline prediction scores",
        manifest_sha256=file_sha256(args.manifest),
        auxiliary_source_sha256=source_hashes,
        protocol="all chronological frames, including training; fixed SMPL vertex indices",
        coordinate_frame="COLMAP world / median training alignment scale; proxy meters",
        units=dict(mae="proxy_m / frame_interval^2", msj="proxy_m^2 / frame_interval^6"),
        formula_source="https://arxiv.org/html/2606.31981v2#S4.SS1 (equations 11 and 12)",
        caveats=[
            "FPS unknown; no seconds-based or cross-capture physical comparison",
            "Raw ROMP meshes combined with supplied alignment; not optimized-SMPL forward",
            "Fit, camera and scale noise contribute; no ground-truth motion error is measured",
            "Lower derivatives alone do not establish motion accuracy (a static avatar is smooth)",
        ],
        scenes=scenes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "auxiliary_source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
