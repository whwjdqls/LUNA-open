"""Acquire only missing native LHM++ priors, with pinned revision and hashes."""

import hashlib
import json
import os
import socket
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    work = Path("/scratch2/whwjdqls99/LUNA-open")
    prior = work / "assets/lhmpp-priors"
    revision = "b683c8f68bede4f318b0bb539730b8e6711d30a0"
    records = []
    for name in ("dense_sample_points/1_160000.ply", "voxel_grid/cano_1_volume.npz"):
        path = Path(hf_hub_download("3DAIGC/LHMPP-Prior", name, revision=revision, local_dir=prior))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        records.append(dict(file=name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
        print(records[-1], flush=True)
    runtime = work / "baselines/lhmpp-20260928/source/pretrained_models"
    shared = work / "assets/lhm-priors/pretrained_models"
    if runtime.is_symlink():
        runtime.unlink()
    runtime.mkdir(exist_ok=True)
    for source in shared.iterdir():
        if source.name in ("voxel_grid", "dense_sample_points"):
            destination = runtime / source.name
            destination.mkdir(exist_ok=True)
            for asset in source.iterdir():
                target = destination / asset.name
                if not target.exists():
                    target.symlink_to(asset)
        elif not (runtime / source.name).exists():
            (runtime / source.name).symlink_to(source, target_is_directory=source.is_dir())
    for row in records:
        target = runtime / row["file"]
        target.parent.mkdir(exist_ok=True, parents=True)
        if not target.exists():
            target.symlink_to(prior / row["file"])
    receipt = dict(
        repo="3DAIGC/LHMPP-Prior",
        revision=revision,
        files=records,
        host=socket.gethostname(),
        job=os.environ["SLURM_JOB_ID"],
        shared_prior=str(shared),
        runtime=str(runtime),
    )
    (prior / "acquisition.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (runtime.parent.parent / "asset-links.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # Hash the 4.5 GB reconstruction checkpoint once on the preparation node.
    # The inference adapter reuses this digest only while size/mtime agree.
    weights = work / "assets/lhmpp_700m/model.safetensors"
    stat = weights.stat()
    fingerprint = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    digest_receipt = weights.parent / "model-verified-sha256.json"
    cached = json.loads(digest_receipt.read_text()) if digest_receipt.exists() else {}
    if cached.get("fingerprint") != fingerprint:
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        after = weights.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Checkpoint changed during hashing")
        verified = dict(path=str(weights), fingerprint=fingerprint, sha256=digest.hexdigest(),
                        host=socket.gethostname(), job=os.environ["SLURM_JOB_ID"])
        digest_receipt.write_text(json.dumps(verified, indent=2) + "\n")


if __name__ == "__main__":
    main()
