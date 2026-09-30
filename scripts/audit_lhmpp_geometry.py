"""Exercise native LHM++ body assets and the full released query set on CPU.

This covers BaseSkinning and its diffused weight volume. The full skinning
wrapper requires CUDA, and this audit does not replace that runtime gate.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import smplx
import torch
from pytorch3d.io import load_ply

from luna_open.provenance import file_sha256

REVISION = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("reference", "checkpoint", "runtime", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError("Preserve prior audits; choose a new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        status="running",
        stages=[],
        limitation="CPU body/weight-volume audit; no full skinning wrapper or checkpoint inference",
    )

    def record(stage, **values):
        report["stages"].append(dict(stage=stage, seconds=time.perf_counter() - started, **values))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["stages"][-1]), flush=True)

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise ValueError("LHM++ source revision changed")
        subprocess.run(["git", "-C", str(args.reference), "diff", "--quiet", "HEAD"], check=True)
        runtime_path = args.runtime / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        for relative, receipt in runtime["files"].items():
            if file_sha256(args.runtime / relative) != receipt["sha256"]:
                raise ValueError(f"Native runtime asset changed: {relative}")
        config_path = args.checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        report.update(
            source_revision=revision,
            runtime_sha256=file_sha256(runtime_path),
            config_sha256=file_sha256(config_path),
            torch=torch.__version__,
            numpy=np.__version__,
        )
        os.chdir(args.runtime)
        sys.path.insert(0, str(args.reference))
        from core.models.rendering.skinnings.base_skinning import BaseSkinning
        from core.structures.voxel_structure import CanoBlendWeightVolume

        base = BaseSkinning(
            human_model_path=config["human_model_path"],
            shape_param_dim=config["shape_param_dim"],
            expr_param_dim=config["expr_param_dim"],
            subdivide_num=config["smplx_subdivide_num"],
            cano_pose_type=config["cano_pose_type"],
        )
        assert (base.vertex_num, base.joint_num) == (10475, 55)
        with Path(config["human_model_path"], "flame/FLAME_NEUTRAL.pkl").open("rb") as stream:
            flame = pickle.load(stream, encoding="latin1")
        expected_expression = torch.from_numpy(flame["shapedirs"][:, :, 300:400]).float()
        torch.manual_seed(19)
        # Canonical and nonzero native SMPL-X inputs, not converted NeuMan fits.
        parameters = dict(
            betas=torch.cat((torch.zeros(1, 10), torch.randn(1, 10) * 0.1)),
            expression=torch.cat((torch.zeros(1, 100), torch.randn(1, 100) * 0.05)),
            body_pose=base.neutral_body_pose.flatten()[None].repeat(2, 1),
            jaw_pose=base.neutral_jaw_pose[None].repeat(2, 1),
            global_orient=torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.2]]),
            left_hand_pose=torch.zeros(2, 45),
            right_hand_pose=torch.zeros(2, 45),
            leye_pose=torch.zeros(2, 3),
            reye_pose=torch.zeros(2, 3),
            transl=torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 1.0]]),
        )
        parameters["body_pose"][1] += torch.randn(63) * 0.05
        for gender, native in base.layer.items():
            torch.testing.assert_close(
                native.expr_dirs[base.face_vertex_idx], expected_expression, rtol=0, atol=0
            )
            standard = smplx.create(
                config["human_model_path"],
                "smplx",
                gender=gender,
                num_betas=10,
                num_expression_coeffs=100,
                use_pca=False,
                use_face_contour=True,
                flat_hand_mean=True,
            )
            # Native LHM++ intentionally transfers FLAME expression bases.
            standard.expr_dirs[base.face_vertex_idx] = expected_expression
            with torch.no_grad():
                actual = native(**parameters).vertices
                expected = standard(**parameters).vertices
            assert actual.shape == (2, 10475, 3) and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
            record(
                "body_forward",
                gender=gender,
                expression_transfer_exact=True,
                max_vertex_error_m=float((actual - expected).abs().max()),
            )
        record(
            "base_skinning",
            expression_vertices=len(base.expr_vertex_idx),
            constraint_vertices=len(base.constrain_body_vertex_idx),
            subdivided_vertices=base.vertex_num_upsampled,
            subdivided_faces=len(base.face_upsampled),
        )
        query_path = Path("pretrained_models/dense_sample_points/1_160000.ply")
        points, _ = load_ply(query_path)
        assert points.shape == (config["dense_sample_pts"], 3) == (160000, 3)
        assert torch.isfinite(points).all()
        volume = CanoBlendWeightVolume("pretrained_models/voxel_grid/cano_1_volume.npz")
        with torch.no_grad():
            weights = volume(points[None])
        assert weights.shape == (1, 160000, 55) and torch.isfinite(weights).all()
        sums = weights.sum(-1)
        within_bounds = (
            (points >= volume.volume_bounds[0]) & (points <= volume.volume_bounds[1])
        ).all(-1)
        record(
            "full_query_weight_volume",
            points=len(points),
            volume_shape=list(volume.diff_weight_volume.shape),
            minimum_weight=float(weights.min()),
            maximum_weight=float(weights.max()),
            weight_sum_min=float(sums.min()),
            weight_sum_max=float(sums.max()),
            weight_sum_max_error=float((sums - 1).abs().max()),
            out_of_bounds_points=int((~within_bounds).sum()),
            volume_bounds=volume.volume_bounds.tolist(),
            query_bounds=torch.stack((points.amin(0), points.amax(0))).tolist(),
        )
        assert float(weights.min()) >= -1e-6
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-3, rtol=0)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
