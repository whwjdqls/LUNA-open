"""Check completed CLI training logs/checkpoints and summarize real run evidence."""

import argparse
import json
import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Checkpoint audits must run on a Slurm compute node")
    config = yaml.safe_load(args.config.read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())
    expected_validation = {
        (scene, frame)
        for scene, info in manifest["scenes"].items()
        for frame in info["splits"]["val"]
    }
    updates = config["training"]["updates"]
    root = Path(config["output"])
    result = {}
    for stage in ("identity", "animator"):
        folder = root / stage
        records = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        if [row["update"] for row in records] != list(range(1, updates + 1)):
            raise ValueError(f"Incomplete or duplicate training updates: {stage}")
        if not all(math.isfinite(value) for row in records for value in row.values()):
            raise ValueError(f"Nonfinite training metric: {stage}")
        validations = [row for row in records if "validation_lpips" in row]
        if not validations:
            raise ValueError(f"Missing validation results: {stage}")
        latest_path = folder / "latest.pt"
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if latest["update"] != updates or latest["config"] != config or latest["stage"] != stage:
            raise ValueError(f"Incomplete or mismatched latest checkpoint: {stage}")
        if latest["scheduler"]["last_epoch"] != updates:
            raise ValueError(f"Incomplete scheduler continuation: {stage}")
        optimizer_steps = [float(value["step"]) for value in latest["optimizer"]["state"].values()]
        if not optimizer_steps or max(optimizer_steps) != updates:
            raise ValueError(f"Incomplete optimizer continuation: {stage}")
        if set(latest["rng"]) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError(f"Incomplete RNG checkpoint: {stage}")
        if latest["manifest_sha256"] != file_sha256(config["manifest"]):
            raise ValueError(f"Wrong manifest: {stage}")
        if latest["smpl_asset_sha256"] != file_sha256(config["smpl_model"]):
            raise ValueError(f"Wrong SMPL asset: {stage}")
        expected_best = min(row["validation_lpips"] for row in validations)
        if latest["best_lpips"] != expected_best:
            raise ValueError(f"Best metric disagrees with the training log: {stage}")
        del latest
        best_path = folder / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if (
            best["best_lpips"] != expected_best
            or best["config"] != config
            or best["stage"] != stage
        ):
            raise ValueError(f"Best checkpoint mismatch: {stage}")
        validation = json.loads((folder / "val-metrics.json").read_text())
        if Path(validation["checkpoint"]).resolve() != best_path.resolve():
            raise ValueError(f"Validation did not use the selected checkpoint: {stage}")
        actual_validation = [(row["scene"], row["frame"]) for row in validation["frames"]]
        if set(actual_validation) != expected_validation or len(actual_validation) != len(
            expected_validation
        ):
            raise ValueError(f"Validation frame membership differs from the manifest: {stage}")
        scores = validation["mean_over_scenes"]
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Nonfinite held-out metric: {stage}")
        # CUDA rasterization is not bitwise deterministic. Record the repeated
        # metric, rather than treating it as an exact-trajectory resume test.
        result[stage] = dict(
            completed_updates=updates,
            best_update=best["update"],
            best_logged_validation_lpips=expected_best,
            validation_frames=len(actual_validation),
            repeated_validation_metrics=scores,
            validation_lpips_difference=scores["lpips"] - expected_best,
            first_update=records[0],
            last_update=records[-1],
            latest_checkpoint=str(latest_path),
            best_checkpoint=str(best_path),
        )
        del best
    report = dict(
        completed_at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        config=config,
        config_sha256=file_sha256(args.config),
        stages=result,
        status="completed stage updates, checkpoint state and validation records checked",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
