"""Freeze report inputs and copy scientific media on a CPU compute node."""

import gc
import importlib.util
import json
import os
import shutil
import socket
from datetime import datetime
from pathlib import Path

import torch

from luna_open.provenance import file_sha256

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
REPO = Path("/home/whwjdqls99/LUNA-open")
OUT = WORK / "reports/luna-progress-20260928"
PRIVATE = WORK / "reports/luna-progress-20260928-work"


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use a compute allocation")
    OUT.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    for name in (
        "qualitative",
        "quantitative/raw",
        "quantitative/charts",
        "evidence/docs",
        "evidence/configs",
        "evidence/audits",
        "evidence/source",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((WORK / "data/neuman/manifest-v2.json").read_text())
    receipt = dict(
        created_at=datetime.now().astimezone().isoformat(),
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        models={},
        sources=[],
        splits={
            s: {k: len(v) for k, v in data["splits"].items()}
            for s, data in manifest["scenes"].items()
        },
    )
    snapshots = {}
    for label, path in [
        ("original_identity", WORK / "runs/neuman/identity/best.pt"),
        ("retrained_identity", WORK / "runs/neuman-identity-v2-20260928/identity/best.pt"),
        ("original_animator", WORK / "runs/neuman/animator/best.pt"),
    ]:
        # Hash and deserialize one opened inode, even if best.pt is replaced.
        import hashlib

        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=False)
        keep = {
            key: state[key]
            for key in (
                "config",
                "stage",
                "update",
                "manifest_sha256",
                "smpl_asset_sha256",
                "feature_metadata",
                "identity",
                "animator",
            )
            if key in state
        }
        destination = PRIVATE / f"{label}.pt"
        torch.save(keep, destination)
        snapshots[label] = str(destination)
        receipt["models"][label] = dict(
            source=str(path),
            source_sha256=checksum,
            update=state["update"],
            snapshot_sha256=file_sha256(destination),
            snapshot=str(destination),
            config=state["config"],
            best_lpips=state.get("best_lpips"),
            stage=state["stage"],
        )
        del state, keep
        gc.collect()
    (PRIVATE / "snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    for stage in ("identity", "animator"):
        src = WORK / "runs/neuman" / stage
        dest = OUT / "quantitative/raw" / f"original_{stage}"
        dest.mkdir()
        for path in sorted(src.glob("*.json")) + [src / "train.jsonl"]:
            shutil.copy2(path, dest / path.name)
    src = WORK / "runs/neuman-identity-v2-20260928/identity"
    dest = OUT / "quantitative/raw/retrained_identity"
    dest.mkdir()
    for path in sorted(src.glob("val-*.json")) + [src / "train.jsonl"]:
        shutil.copy2(path, dest / path.name)
    rows = [
        json.loads(line) for line in (dest / "train.jsonl").read_text().splitlines() if line.strip()
    ]
    receipt["training_update_at_snapshot"] = rows[-1]["update"]
    receipt["available_modules"] = {
        name: importlib.util.find_spec(name) is not None
        for name in ("matplotlib", "markdown", "markdown2", "pptx")
    }
    media_sources = [
        ("01_original_identity_all_frames", WORK / "outputs/gpu-2336972/identity-10000-train-test"),
        (
            "02_original_identity_gt_canonical_lbs_gifs",
            WORK / "outputs/gpu-2336972/identity-10000-train-test-gifs",
        ),
        ("03_neural_animator_5000_failure", WORK / "outputs/gpu-2336972/qualitative-animator-5000"),
        (
            "04_canonical_and_neural_animation_5000",
            WORK / "outputs/gpu-2336972/identity-and-animation-5000",
        ),
        ("05_identity_diagnosis", WORK / "outputs/gpu-2343414/identity-baseline-diagnosis-v2"),
        ("06_animator_normalization_probe", WORK / "outputs/gpu-2336972/normalization-probe-5000"),
        ("07_animator_fresh_local_probe", WORK / "outputs/gpu-2336972/fresh-local-probe-5000"),
        ("08_smoke_example", WORK / "outputs/gpu-2336972/network-smoke"),
    ]
    for label, source in media_sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = OUT / "qualitative" / label
        count = 0
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".mp4",
                ".json",
            }:
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                count += 1
        receipt["sources"].append(
            dict(original=str(source), copied_to=str(destination.relative_to(OUT)), files=count)
        )
    destination = OUT / "qualitative/09_retraining_validation_progress"
    destination.mkdir()
    selected = {
        250,
        1000,
        3000,
        5000,
        7500,
        10000,
        receipt["models"]["retrained_identity"]["update"],
    }
    val_files = sorted(src.glob("val-*.jpg"))
    selected.add(int(val_files[-1].stem.split("-")[-1]))
    for path in val_files:
        if int(path.stem.split("-")[-1]) in selected:
            shutil.copy2(path, destination / path.name)
    for path in sorted((REPO / "docs").glob("*.md")):
        shutil.copy2(path, OUT / "evidence/docs" / path.name)
    for name in (
        "neuman_yonsei.yaml",
        "neuman_yonsei_identity_v2.yaml",
        "neuman_yonsei_smoke.yaml",
    ):
        shutil.copy2(REPO / "configs" / name, OUT / "evidence/configs" / name)
    shutil.copy2(WORK / "data/neuman/manifest-v2.json", OUT / "evidence/manifest-v2.json")
    for path in (WORK / "outputs/gpu-2336972").glob("*.json"):
        if any(key in path.name for key in ("audit", "10000", "diagnos")):
            shutil.copy2(path, OUT / "evidence/audits" / path.name)
    for path in (WORK / "runs/neuman").glob("*audit*.json"):
        shutil.copy2(path, OUT / "evidence/audits" / path.name)
    for directory, prefix in [
        (WORK / "outputs/gpu-2343414/identity-attention-all-keys", "attention-"),
        (WORK / "outputs/gpu-2344049/identity-batching/results", "batching-"),
    ]:
        for path in directory.glob("*.json"):
            shutil.copy2(path, OUT / "evidence/audits" / (prefix + path.name))
    for path in (WORK / "runs/neuman-identity-v2-20260928/provenance").glob("*.json"):
        shutil.copy2(path, OUT / "evidence/audits" / ("identity-v2-" + path.name))
    receipt["copied_file_count"] = sum(1 for p in OUT.rglob("*") if p.is_file())
    (OUT / "evidence/report-snapshot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("sources", "splits", "models")}, indent=2
        )
    )
    print(
        json.dumps(
            {
                key: {k: v for k, v in value.items() if k not in ("config",)}
                for key, value in receipt["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
