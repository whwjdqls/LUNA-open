"""Measure NeuMan coverage of LUNA equation 3's bounded translation head.

The implementation follows the paper literally: mean + std * tanh(raw).
This audit measures the root-translation range, not an error lower bound for
the complete animator: its local residuals can also move Gaussian centers.
Only training statistics determine the bounds; test frames are not inspected.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from luna_open.data.neuman import NeuManDataset, frame_annotation
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.smpl import SMPLTeacher
from luna_open.training import translation_statistics


def summarize(rows):
    translations = torch.tensor([r["translation_m"] for r in rows])
    outside = torch.tensor([r["outside_axes"] for r in rows])
    errors = torch.tensor([r["distance_to_closed_box_m"] for r in rows])
    return dict(
        frames=len(rows),
        outside_any_count=int(outside.any(-1).sum()),
        outside_any_fraction=float(outside.any(-1).float().mean()),
        outside_axis_counts=outside.sum(0).tolist(),
        translation_min_m=translations.amin(0).tolist(),
        translation_max_m=translations.amax(0).tolist(),
        distance_to_closed_box_mean_m=float(errors.mean()),
        distance_to_closed_box_p95_m=float(torch.quantile(errors, 0.95)),
        distance_to_closed_box_max_m=float(errors.max()),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).eval()
    train = NeuManDataset(cfg["data_root"], cfg["manifest"], split="train")
    mean, std = translation_statistics(teacher, train)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid training translation statistics")
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    if state["manifest_sha256"] != file_sha256(cfg["manifest"]):
        raise ValueError("Checkpoint and audit data differ")
    if state["config"].get("smpl_pose_blend_shapes", True) != teacher.pose_blend_shapes:
        raise ValueError("Checkpoint teacher convention differs")
    differences = {}
    for key, value in (("translation_mean", mean), ("translation_std", std)):
        saved = state["animator"][key]
        torch.testing.assert_close(saved, value, atol=2e-6, rtol=2e-6)
        differences[key] = float((saved - value).abs().max())
    del state
    lower, upper = mean - std, mean + std
    records, summaries = [], {}
    for split in ("train", "val"):
        data = NeuManDataset(cfg["data_root"], cfg["manifest"], split=split)
        rows = []
        for scene, name in data.items:
            annotation = frame_annotation(data.root / scene, data.lookup[scene][name])
            pose, betas, transform = [
                torch.from_numpy(annotation[k])[None] for k in ("pose", "betas", "body_to_camera")
            ]
            _, translation = teacher.global_motion(pose, betas, transform)
            translation = translation[0]
            if not torch.isfinite(translation).all():
                raise ValueError(f"Invalid translation: {scene}/{name}")
            nearest = translation.clamp(min=lower, max=upper)
            rows.append(
                dict(
                    split=split,
                    scene=scene,
                    frame=name,
                    translation_m=translation.tolist(),
                    normalized_translation=((translation - mean) / std).tolist(),
                    outside_axes=((translation < lower) | (translation > upper)).tolist(),
                    distance_to_closed_box_m=float((translation - nearest).norm()),
                )
            )
        summaries[split] = dict(
            overall=summarize(rows),
            per_scene={
                scene: summarize([r for r in rows if r["scene"] == scene])
                for scene in data.metadata
            },
        )
        records.extend(rows)
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        config=str(args.config),
        manifest_sha256=file_sha256(cfg["manifest"]),
        smpl_asset_sha256=file_sha256(cfg["smpl_model"]),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint),
        mean_m=mean.tolist(),
        std_m=std.tolist(),
        lower_m=lower.tolist(),
        upper_m=upper.tolist(),
        checkpoint_stat_max_abs_differences=differences,
        statistics_sampling=f"Frame-uniform over all {len(train)} training frames; population std",
        source="https://arxiv.org/html/2606.31981v2#S3.SS2",
        summaries=summaries,
        frames=records,
        limitations=[
            "Closed-box distance is optimistic: tanh reaches its endpoints only in the limit",
            "Measures annotated global roots, not complete-model reconstruction error",
            "Unbounded local position residuals can absorb a common translation offset",
            "No trained predictor, image metrics or test-frame measurements in this audit",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
