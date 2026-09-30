"""Download the frozen DNA release one file at a time through authenticated rclone.

Run on a CPU allocation. Files, logs, Drive IDs and receipts remain in private
storage outside the repository. Each selected file gets one attempt. A file's
download quota does not prevent checking other files; API rate errors stop the
batch because they affect the client. No remote writes or quota workarounds.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from acquire_dna import now, write_json


def local_path(root, name):
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes private data root")
    return path


def verify_file(path, entry, previous, source_md5):
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("File size differs from the frozen inventory")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        magic = handle.read(8)
        if path.suffix == ".smc" and magic != b"\x89HDF\r\n\x1a\n":
            raise ValueError("SMC does not have an HDF5 signature")
        handle.seek(0)
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(chunk)
            md5.update(chunk)
    if previous.get("sha256") and sha.hexdigest() != previous["sha256"]:
        raise ValueError("File differs from its saved SHA256 receipt")
    if source_md5 and md5.hexdigest() != source_md5:
        raise ValueError("File differs from authenticated Drive MD5")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP member failed CRC verification")
    if path.suffix == ".json":
        json.loads(path.read_text())
    return dict(
        status="complete",
        bytes=entry["bytes"],
        sha256=sha.hexdigest(),
        md5=md5.hexdigest(),
        source_md5_verified=bool(source_md5),
        verified_at=now(),
    )


def priority(entry):
    name = entry["path"]
    if entry["bytes"] < 50 * 1024**2:
        return (0, "", entry["bytes"])
    if "/dna_rendering_part1_annotations/" in name or "/dna_rendering_part1_main/" in name:
        return (1, Path(name).stem.removesuffix("_annots"), int("_annots" not in name))
    return (2, "", entry["bytes"])


def run(args):
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path = root / "acquisition.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["manifest_sha256"] != manifest_sha:
        raise ValueError("Receipt and inventory differ")
    expected = {x["path"]: x for x in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate inventory paths")
    for name in expected:
        local_path(root, name)
    if not set(args.skip) <= set(expected):
        raise ValueError("Skipped file is absent from the inventory")
    stamp = now().replace(":", "-")
    log_root = root / "rclone-batches" / stamp
    log_root.mkdir(parents=True)

    # Verify current authenticated metadata before transferring any file.
    listing_path = log_root / "listing.json"
    with listing_path.open("w") as output, (log_root / "listing.log").open("w") as error:
        result = subprocess.run(
            [
                args.rclone,
                "lsjson",
                args.remote,
                "--recursive",
                "--files-only",
                "--hash",
                "--tpslimit",
                "1",
                "--retries",
                "1",
                "--low-level-retries",
                "1",
            ],
            stdout=output,
            stderr=error,
            timeout=300,
        )
    if result.returncode:
        raise RuntimeError(f"Authenticated listing failed; see {log_root / 'listing.log'}")
    listing = json.loads(listing_path.read_text())
    current = {"part1/" + x["Path"]: x for x in listing}
    pinned = {
        "part1/" + x["Path"]: x
        for x in json.loads((root / "authenticated-listing.json").read_text())
    }
    part1 = {name for name in expected if name.startswith("part1/")}
    if len(current) != len(listing) or set(current) != part1:
        raise ValueError("Authenticated folder membership differs from the inventory")
    for name, row in current.items():
        if (
            row["ID"] != expected[name]["id"]
            or row["Size"] != expected[name]["bytes"]
            or not row.get("Hashes", {}).get("md5")
            or row["Hashes"]["md5"] != pinned[name]["Hashes"]["md5"]
        ):
            raise ValueError("Authenticated file metadata/checksum changed")
    required = sum(
        x["bytes"]
        for name, x in expected.items()
        if name not in args.skip and not (root / name).exists()
    )
    if shutil.disk_usage(root).free < required + 20 * 1024**3:
        raise ValueError("Insufficient space for selected files plus 20 GiB reserve")
    batch = dict(
        started_at=now(),
        manifest_sha256=manifest_sha,
        attempts=[],
        skipped=args.skip,
        state="running",
    )
    write_json(log_root / "batch.json", batch)
    for index, entry in enumerate(sorted(manifest["files"], key=priority)):
        name = entry["path"]
        if name in args.skip:
            continue
        path = local_path(root, name)
        previous = receipt["files"].get(name, {})
        attempt = dict(path=name, started_at=now(), transport="authenticated rclone serial")
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                log_path = log_root / f"file-{index:03d}.log"
                with log_path.open("w") as output:
                    result = subprocess.run(
                        [
                            args.rclone,
                            "backend",
                            "copyid",
                            args.remote,
                            entry["id"],
                            str(path),
                            "--checksum",
                            "--immutable",
                            "--transfers",
                            "1",
                            "--multi-thread-streams",
                            "0",
                            "--tpslimit",
                            "1",
                            "--retries",
                            "1",
                            "--low-level-retries",
                            "1",
                            "--drive-stop-on-download-limit",
                            "--stats",
                            "30s",
                            "--stats-one-line",
                            "-v",
                        ],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )
                output = log_path.read_text()
                attempt.update(exit_code=result.returncode, log=str(log_path.relative_to(root)))
                if result.returncode:
                    attempt["status"] = (
                        "api_rate_limited"
                        if "rateLimitExceeded" in output or "RATE_LIMIT_EXCEEDED" in output
                        else "quota_exceeded"
                        if "downloadQuotaExceeded" in output
                        else "error"
                    )
            if "status" not in attempt:
                attempt.update(
                    verify_file(
                        path, entry, previous, current.get(name, {}).get("Hashes", {}).get("md5")
                    )
                )
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            attempt.update(status="error", error=str(error))
        attempt["finished_at"] = now()
        batch["attempts"].append(attempt)
        saved = {**previous, **attempt}
        history = list(previous.get("serial_attempts", []))
        history.append(attempt)
        saved["serial_attempts"] = history
        if attempt["status"] == "complete":
            saved.pop("error", None)
        receipt["files"][name] = saved
        receipt["updated_at"] = now()
        write_json(receipt_path, receipt)
        write_json(log_root / "batch.json", batch)
        print(
            json.dumps({key: attempt[key] for key in ("path", "status", "finished_at")}), flush=True
        )
        if attempt["status"] in {"api_rate_limited", "error"}:
            batch["state"] = "stopped_on_error"
            break
    else:
        batch["state"] = "finished_selected_files"
    batch["finished_at"] = now()
    batch["complete"] = sum(x.get("status") == "complete" for x in receipt["files"].values())
    batch["total"] = len(manifest["files"])
    write_json(log_root / "batch.json", batch)
    print(
        json.dumps(dict(complete=batch["complete"], total=batch["total"], batch=str(log_root))),
        flush=True,
    )
    return int(batch["complete"] != batch["total"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remote", default="dna:")
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Inventory paths already attempted in this run"
    )
    args = parser.parse_args()
    if args.root.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Data storage must be outside the repository")
    os.umask(0o077)
    with (args.root / ".acquire.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
