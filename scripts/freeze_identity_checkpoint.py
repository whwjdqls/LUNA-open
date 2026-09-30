"""Make an immutable inference snapshot of one atomically replaced training file."""

import argparse
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use a Slurm compute node")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=False)
    kept = {
        key: state[key]
        for key in (
            "config",
            "stage",
            "update",
            "manifest_sha256",
            "smpl_asset_sha256",
            "feature_metadata",
            "identity",
        )
    }
    destination = args.output / f"identity-{state['update']:06d}.pt"
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(kept, destination)
    with destination.open("rb") as handle:
        snapshot_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = dict(
        source=str(args.source),
        source_sha256=source_hash,
        snapshot=str(destination),
        snapshot_sha256=snapshot_hash,
        update=state["update"],
        selection=args.selection,
        best_lpips=state.get("best_lpips"),
        created_at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
    )
    destination.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
