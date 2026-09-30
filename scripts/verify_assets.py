"""Verify pinned downloaded files against Hugging Face repository metadata.

Run hashing on a Slurm CPU allocation. This checks bytes and acquisition
provenance without deserializing model files or asserting runtime compatibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/assets.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("assets", nargs="+")
    args = parser.parse_args()
    catalog = yaml.safe_load(args.catalog.read_text())
    api, reports = HfApi(), {}
    for name in args.assets:
        spec = catalog[name]
        receipt = json.loads((args.root / f"{name}-receipt.json").read_text())
        if receipt.get("status") != "downloaded" or any(
            receipt.get(key) != spec[key] for key in ("repo", "revision", "files")
        ):
            raise ValueError(f"Download receipt disagrees with the pinned catalog: {name}")
        info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise ValueError(f"Expected a full immutable commit pin: {name}")
        metadata = {file.rfilename: file for file in info.siblings}
        files = {}
        for relative in spec["files"]:
            remote = metadata[relative]
            path = args.root / name / relative
            size = path.stat().st_size
            if size != remote.size:
                raise ValueError(f"File length differs from pinned repository: {path}")
            sha256 = file_sha256(path)
            if remote.lfs is not None:
                expected = remote.lfs.sha256
                if sha256 != expected:
                    raise ValueError(f"LFS SHA256 differs: {path}")
                verification = dict(kind="lfs-sha256", expected=expected)
            else:
                # Ordinary Git objects hash the header and payload together.
                content = path.read_bytes()
                actual = hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest()
                if actual != remote.blob_id:
                    raise ValueError(f"Git blob SHA1 differs: {path}")
                verification = dict(kind="git-blob-sha1", expected=remote.blob_id)
            files[relative] = dict(bytes=size, sha256=sha256, verification=verification)
        reports[name] = dict(
            repo=spec["repo"],
            revision=info.sha,
            files=files,
            count=len(files),
            total_bytes=sum(f["bytes"] for f in files.values()),
        )
        print(name, "verified", reports[name]["count"], "files", flush=True)
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        assets=reports,
        limitation="Byte integrity only; model compatibility and output quality are separate checks",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
