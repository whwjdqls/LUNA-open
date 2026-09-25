#!/usr/bin/env python3
"""Acquire Apple's public archive, check every ZIP CRC, and extract safely.

Stdlib only. The locally calculated SHA256 records content identity; Apple does
not publish an independently verified SHA256 in the upstream download script.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip"
EXPECTED_BYTES = 2154631653
EXPECTED_SHA256 = "3eec31be4fb4bbb95509db08e5956a93df76a9141a985e31733e883fb7e404c3"


def acquire(root: Path, extract: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "dataset.zip"
    if not archive.exists():
        partial = root / "dataset.zip.part"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--retry-delay",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                URL,
            ],
            check=True,
        )
        if partial.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Archive size changed; review upstream before extraction")
        partial.rename(archive)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"Unexpected archive size: {archive.stat().st_size}; preserve and inspect it"
        )
    print("Computing archive SHA256...", flush=True)
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "Archive content differs from the verified 2026-09-25 copy; preserve and review it"
        )
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        members = []
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink: {info.filename}")
            if path.parts[0] == "dataset" and not any(p.startswith(".") for p in path.parts):
                members.append(info)
        required = sum(m.file_size for m in members)
        if extract:
            if shutil.disk_usage(root).free < required:
                raise OSError("Insufficient filesystem space; also check the shared project quota")
            print(f"Extracting {len(members)} entries ({required:,} bytes)...", flush=True)
            for m in members:
                z.extract(m, root)
    result = dict(
        url=URL,
        bytes=archive.stat().st_size,
        sha256=digest,
        zip_crc="all passed",
        sha256_provenance="computed locally, not an upstream-published checksum",
        extracted=extract,
        extracted_bytes=required if extract else 0,
        verified_at=datetime.now(timezone.utc).isoformat(),
    )
    (root / "acquisition.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    acquire(args.root, not args.verify_only)
