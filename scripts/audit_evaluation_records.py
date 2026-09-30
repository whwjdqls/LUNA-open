"""Audit saved validation/test metrics on a Slurm compute node, without inference."""

import argparse
import hashlib
import json
import math
import os
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

METRICS = {"psnr", "l1", "ssim", "mask_iou", "lpips"}
PROTOCOL = "seen-sequence, frame-held-out, annotated crops"


def signature(stat):
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def stable_digest(path):
    """Refuse a file replaced or modified while it is hashed."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if signature(before) != signature(path.stat()):
        raise ValueError(f"File changed during hashing: {path}")
    return digest.hexdigest()


def stable_text(path):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    if signature(before) != signature(path.stat()):
        raise ValueError(f"File changed during reading: {path}")
    return raw.decode(), hashlib.sha256(raw).hexdigest()


def check_scores(scores, label):
    if not isinstance(scores, dict) or set(scores) != METRICS:
        raise ValueError(f"Wrong metric schema: {label}")
    if not all(type(value) in (int, float) and math.isfinite(value) for value in scores.values()):
        raise ValueError(f"Nonfinite or nonnumeric metric: {label}")


def inspect_evaluation(report, manifest, split, checkpoint, manifest_sha256):
    """Check membership and independently recompute scene means from JSON scores."""
    expected = [
        (scene, frame)
        for scene, info in manifest["scenes"].items()
        for frame in info["splits"][split]
    ]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError(f"Empty or duplicate manifest membership: {split}")
    if report["manifest_sha256"] != manifest_sha256:
        raise ValueError("Evaluation manifest hash mismatch")
    if Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Evaluation did not use the selected best checkpoint path")
    if report["protocol"] != PROTOCOL:
        raise ValueError("Evaluation protocol mismatch")
    actual = [(row["scene"], row["frame"]) for row in report["frames"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f"Evaluation frame membership mismatch: {split}")
    by_scene = defaultdict(list)
    for row in report["frames"]:
        check_scores(row["metrics"], f"{row['scene']}/{row['frame']}")
        by_scene[row["scene"]].append(row["metrics"])
    if set(report["per_scene"]) != set(by_scene):
        raise ValueError("Evaluation scene membership mismatch")
    recomputed = {
        scene: {key: math.fsum(row[key] for row in rows) / len(rows) for key in METRICS}
        for scene, rows in by_scene.items()
    }
    macro = {
        key: math.fsum(row[key] for row in recomputed.values()) / len(recomputed) for key in METRICS
    }
    differences = []
    comparisons = [(report["mean_over_scenes"], macro, "mean_over_scenes")]
    comparisons += [
        (report["per_scene"][scene], scores, scene) for scene, scores in recomputed.items()
    ]
    for saved, calculated, label in comparisons:
        check_scores(saved, label)
        for key in METRICS:
            differences.append(abs(saved[key] - calculated[key]))
            if not math.isclose(saved[key], calculated[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Evaluation aggregation mismatch: {label}/{key}")
    return dict(
        split=split,
        frames=len(actual),
        frames_per_scene=dict(sorted(Counter(scene for scene, _ in actual).items())),
        unique_manifest_membership=True,
        all_frame_and_aggregate_metrics_finite=True,
        aggregation_max_abs_difference=max(differences),
        per_scene=report["per_scene"],
        mean_over_scenes=report["mean_over_scenes"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages", nargs="+", choices=("identity", "animator"), default=["identity", "animator"]
    )
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    node = os.environ.get("SLURMD_NODENAME", "")
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not node
        or socket.gethostname().split(".")[0] != node.split(".")[0]
    ):
        raise RuntimeError("Evaluation audits must run through Slurm on a compute node")
    if args.output.exists():
        raise FileExistsError(f"Preserve existing audit evidence: {args.output}")
    config_text, config_digest = stable_text(args.config)
    config = yaml.safe_load(config_text)
    manifest_text, manifest_digest = stable_text(config["manifest"])
    manifest = json.loads(manifest_text)
    results = {}
    for stage in dict.fromkeys(args.stages):
        folder = Path(config["output"]) / stage
        log_text, log_digest = stable_text(folder / "train.jsonl")
        if not log_text.endswith("\n"):
            raise ValueError(f"Incomplete training log write: {stage}")
        records = [json.loads(line) for line in log_text.splitlines()]
        updates = config["training"]["updates"]
        if [row["update"] for row in records] != list(range(1, updates + 1)):
            raise ValueError(f"Stage has not completed the configured updates: {stage}")
        if not all(
            type(v) in (int, float) and math.isfinite(v) for row in records for v in row.values()
        ):
            raise ValueError(f"Nonfinite or nonnumeric training record: {stage}")
        validations = [row for row in records if "validation_lpips" in row]
        interval = config["training"]["validate_every"]
        expected = [n for n in range(1, updates + 1) if n % interval == 0 or n == updates]
        if [row["update"] for row in validations] != expected:
            raise ValueError(f"Missing or unscheduled validation: {stage}")
        best = min(validations, key=lambda row: row["validation_lpips"])
        checkpoint = folder / "best.pt"
        evaluations = {}
        for split in dict.fromkeys(args.splits):
            path = folder / f"{split}-metrics.json"
            text, digest = stable_text(path)
            result = inspect_evaluation(
                json.loads(text), manifest, split, checkpoint, manifest_digest
            )
            result.update(path=str(path), sha256=digest)
            if split == "val":
                result["lpips_difference_from_best_logged_validation"] = (
                    result["mean_over_scenes"]["lpips"] - best["validation_lpips"]
                )
            evaluations[split] = result
        results[stage] = dict(
            completed_updates=updates,
            train_log_sha256=log_digest,
            best_logged_update=best["update"],
            best_logged_validation_lpips=best["validation_lpips"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=stable_digest(checkpoint),
            evaluations=evaluations,
        )
    report = dict(
        audited_at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        device="cpu",
        source=str(Path(__file__).resolve()),
        source_sha256=stable_digest(__file__),
        config=str(args.config.resolve()),
        config_sha256=config_digest,
        manifest_sha256=manifest_digest,
        stages=results,
        status="Requested saved evaluation records passed integrity and aggregation checks",
        limitations=[
            "Scores were recomputed from JSON records, not rendered images or GPU inference.",
            "Checkpoint bytes were hashed; tensor/state correctness requires the separate checkpoint audit.",
            "Evaluation metadata records a checkpoint path, not the hash of weights used at evaluation time.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
