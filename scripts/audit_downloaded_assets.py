"""Verify each pinned Hugging Face file against its upstream content hash."""

import argparse
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/assets.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    catalog = yaml.safe_load(args.catalog.read_text())
    report = dict(
        verified_at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        assets={},
    )
    api = HfApi()
    failed = False
    for name, spec in catalog.items():
        row = dict(repo=spec["repo"], revision=spec["revision"], files={})
        report["assets"][name] = row
        try:
            info = api.model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
            if info.sha != spec["revision"]:
                raise ValueError("Resolved revision differs from catalog")
            upstream = {f.rfilename: f for f in info.siblings}
            for filename in spec["files"]:
                expected = upstream[filename]
                path = args.root / name / filename
                size = path.stat().st_size
                if size != expected.size:
                    raise ValueError("Downloaded size differs from upstream metadata")
                digest = hashlib.sha256()
                blob = hashlib.sha1(f"blob {size}\0".encode())
                with path.open("rb") as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        digest.update(block)
                        if expected.lfs is None:
                            blob.update(block)
                sha = digest.hexdigest()
                if expected.lfs is not None:
                    if sha != expected.lfs.sha256:
                        raise ValueError("Downloaded SHA256 differs from upstream LFS metadata")
                    verification = "upstream LFS SHA256"
                else:
                    if blob.hexdigest() != expected.blob_id:
                        raise ValueError("Downloaded Git blob hash differs from upstream")
                    verification = "upstream Git blob SHA1; local SHA256 also recorded"
                row["files"][filename] = dict(
                    path=str(path.resolve()), bytes=size, sha256=sha, verification=verification
                )
            row["status"] = "verified"
        except Exception as error:
            failed = True
            row.update(status="failed", error_type=type(error).__name__)
            # Avoid exception text, signed URLs, headers, or credentials in logs.
        print(name, row["status"], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".part")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.report)
    if failed:
        raise SystemExit("Some downloaded assets failed verification; inspect the report")


if __name__ == "__main__":
    main()
