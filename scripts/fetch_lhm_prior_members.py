"""Index an uncompressed upstream tar by HTTP ranges and fetch selected assets.

This avoids downloading already acquired Sapiens weights and unused pose trackers.
The public URL is linked in the pinned LHM README. Run on Slurm compute nodes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar"


def get_range(start, length):
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={start}-{start + length - 1}"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Range request returned {response.status}")
                expected = f"bytes {start}-{start + length - 1}/"
                if not response.headers["Content-Range"].startswith(expected):
                    raise RuntimeError("Unexpected Content-Range")
                payload = response.read()
                if len(payload) != length:
                    raise RuntimeError("Truncated range response")
                return payload
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def scan_tar(destination):
    previous = json.loads(destination.read_text()) if destination.exists() else {}
    rows = previous.get("members", [])
    position = (rows[-1]["offset"] + ((rows[-1]["size"] + 511) // 512) * 512) if rows else 0
    pending_name = None
    while True:
        header = get_range(position, 512)
        if header == bytes(512):
            break
        info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
        content = position + 512
        next_position = content + ((info.size + 511) // 512) * 512
        if info.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            pending_name = get_range(content, info.size).rstrip(b"\0").decode()
        elif info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            raise RuntimeError("PAX archive needs an explicit parser; stopping")
        else:
            name = pending_name or info.name
            pending_name = None
            row = dict(name=name, size=info.size, offset=content, regular=info.isfile())
            rows.append(row)
            print(json.dumps(row), flush=True)
            destination.write_text(
                json.dumps(dict(url=URL, complete=False, members=rows), indent=2)
            )
        position = next_position
    result = dict(url=URL, complete=True, members=rows)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def download_member(row, root):
    relative = PurePosixPath(row["name"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe tar path: {relative}")
    dest = root / relative
    receipt = dest.with_name(dest.name + ".receipt.json")
    if dest.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["source"] == row and dest.stat().st_size == row["size"]:
            return saved
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    chunks = [(i, min(4 * 1024**2, row["size"] - i)) for i in range(0, row["size"], 4 * 1024**2)]
    with partial.open("wb") as handle:
        handle.truncate(row["size"])

        def obtain(chunk):
            offset, length = chunk
            return offset, get_range(row["offset"] + offset, length)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for offset, payload in pool.map(obtain, chunks):
                handle.seek(offset)
                handle.write(payload)
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    partial.replace(dest)
    result = dict(url=URL, source=row, sha256=digest.hexdigest(), bytes=dest.stat().st_size)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Acquired {dest}: {row['size']} bytes", flush=True)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--component", choices=["core", "face"], default="core")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = args.output / "remote-tar-index.json"
    metadata = json.loads(index.read_text()) if index.exists() else {}
    if not metadata.get("complete"):
        metadata = scan_tar(index)
    if args.index_only:
        return
    selected = []
    for row in metadata["members"]:
        name = row["name"]
        wanted = name.endswith(
            (
                "/SMPLX_NEUTRAL.npz",
                "/SMPLX_MALE.npz",
                "/SMPLX_FEMALE.npz",
                "/MANO_SMPLX_vertex_ids.pkl",
                "/SMPL-X__FLAME_vertex_ids.npy",
                "/flame/2019/generic_model.pkl",
                "/flame/FLAME_NEUTRAL.pkl",
                "/flame/flame_static_embedding.pkl",
                "/flame/flame_dynamic_embedding.npy",
                "/voxel_192.pth",
                "/human_prior_constrain.npz",
                "/1_40000.ply",
                "/arcface_resnet18.pth",
            )
        ) or name.endswith(("/LICENSE", "/Readme.pdf", "/version.txt"))
        if args.component == "face":
            wanted = name.endswith(
                ("/RealESRGAN_x4plus.pth", "/parsing_parsenet.pth", "/detection_Resnet50_Final.pth")
            )
        if row["regular"] and wanted:
            selected.append(row)
    print(
        f"Selected {len(selected)} assets, {sum(row['size'] for row in selected)} bytes", flush=True
    )
    receipts = [download_member(row, args.output) for row in selected]
    (args.output / f"selected-{args.component}-acquisition.json").write_text(
        json.dumps(
            dict(
                source=URL,
                host=socket.gethostname(),
                slurm_job=os.environ["SLURM_JOB_ID"],
                files=receipts,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
