"""Audit real training checkpoints/logs after a completed development run.

Checks schedule continuity, checkpoint metadata, finite states, best-validation
selection, and identity freezing during animator training. No quality threshold
or claim of bitwise CUDA trajectory reproducibility is implied.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from luna_open.provenance import file_sha256, validate_body_asset


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_tensors(tree, path="state"):
    if isinstance(tree, torch.Tensor):
        if not torch.isfinite(tree).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {path}")
    elif isinstance(tree, dict):
        for key, value in tree.items():
            finite_tensors(value, f"{path}.{key}")
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            finite_tensors(value, f"{path}.{index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root, options = Path(config["output"]), config["training"]
    manifest = json.loads(Path(config["manifest"]).read_text())
    manifest_hash = file_sha256(config["manifest"])
    body_hash = file_sha256(config["smpl_model"])
    results, identity_best_hash = {}, None
    for stage in ("identity", "animator"):
        folder = root / stage
        rows = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        if [row["update"] for row in rows] != list(range(1, options["updates"] + 1)):
            raise ValueError(f"Missing/duplicated training updates: {stage}")
        for row in rows:
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Nonfinite logged loss: {stage}/{row['update']}")
            expected_lr = options["learning_rate"] * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * row["update"] / options["updates"]))
            )
            if not math.isclose(row["lr"], expected_lr, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Scheduler discontinuity: {stage}/{row['update']}")
            if stage == "animator":
                warmup = row["update"] <= options["global_warmup"]
                if warmup and any(key in row for key in ("rgb", "lpips", "structural")):
                    raise ValueError("Animator local/render losses appeared during global warmup")
                if not warmup and not {
                    "rgb",
                    "mask",
                    "lpips",
                    "structural",
                    "rotation",
                    "projection",
                } <= set(row):
                    raise ValueError("Animator is missing a post-warmup loss")
        validation = [row for row in rows if "validation_lpips" in row]
        best_row = min(validation, key=lambda row: row["validation_lpips"])
        checkpoints = {}
        for name in ("best", "latest"):
            path = folder / f"{name}.pt"
            state = torch.load(path, weights_only=False, map_location="cpu")
            validate_body_asset(state, body_hash)
            if (
                state["stage"] != stage
                or state["config"] != config
                or state["manifest_sha256"] != manifest_hash
            ):
                raise ValueError(f"Checkpoint provenance mismatch: {stage}/{name}")
            expected_update = best_row["update"] if name == "best" else options["updates"]
            if (
                state["update"] != expected_update
                or state["scheduler"]["last_epoch"] != expected_update
            ):
                raise ValueError(f"Wrong checkpoint/scheduler update: {stage}/{name}")
            if not math.isclose(state["best_lpips"], best_row["validation_lpips"], abs_tol=1e-10):
                raise ValueError("Checkpoint selection does not match validation log")
            for kind in state["feature_metadata"]:
                current = json.loads(
                    (Path(config["features"]) / kind / "metadata.json").read_text()
                )
                if current != state["feature_metadata"][kind]:
                    raise ValueError(f"Checkpoint feature provenance differs: {kind}")
            for key in ("identity", "animator", "optimizer"):
                finite_tensors(state[key], key)
            identity_hash = state_hash(state["identity"])
            if stage == "identity" and name == "best":
                identity_best_hash = identity_hash
            if stage == "animator" and identity_hash != identity_best_hash:
                raise ValueError("Animator training changed its frozen identity checkpoint")
            steps = [int(value["step"]) for value in state["optimizer"]["state"].values()]
            checkpoints[name] = dict(
                update=state["update"],
                best_lpips=state["best_lpips"],
                file_sha256=file_sha256(path),
                bytes=path.stat().st_size,
                identity_state_sha256=identity_hash,
                optimizer_step_range=[min(steps), max(steps)],
                scheduler_epoch=state["scheduler"]["last_epoch"],
            )
            del state
        results[stage] = dict(
            updates=len(rows),
            validation_updates=[row["update"] for row in validation],
            checkpoints=checkpoints,
        )
        print(json.dumps({stage: results[stage]}), flush=True)
    scores = json.loads((root / "animator/val-metrics.json").read_text())
    expected = {
        (scene, frame)
        for scene, info in manifest["scenes"].items()
        for frame in info["splits"]["val"]
    }
    actual = {(row["scene"], row["frame"]) for row in scores["frames"]}
    if actual != expected or len(scores["frames"]) != len(expected):
        raise ValueError("Validation output membership differs from protocol")
    if scores["manifest_sha256"] != manifest_hash:
        raise ValueError("Validation manifest mismatch")
    result = dict(
        config=str(args.config),
        manifest_sha256=manifest_hash,
        smpl_asset_sha256=body_hash,
        stages=results,
        frozen_identity_preserved=True,
        validation_frames=len(expected),
        validation_metrics=scores["mean_over_scenes"],
        limitations="State/schedule/integration audit; short smoke metrics are not learned-avatar quality evidence",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
