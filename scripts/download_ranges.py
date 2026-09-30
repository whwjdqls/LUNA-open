"""Resume a public download using validated HTTP ranges.

For servers advertising Accept-Ranges: bytes. Run the dataset/asset's checksum
verification after this transport step; byte counts alone do not verify content.
"""

import argparse
import concurrent.futures
import fcntl
import shutil
import time
import urllib.request
from pathlib import Path


def download(url: str, target: Path, size: int, workers: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            if target.stat().st_size != size:
                raise ValueError("Existing target has the wrong size; preserve and inspect it")
            print(f"Already present; checksum verification still required: {target}", flush=True)
            return
        partial = target.with_name(target.name + ".part")
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > size:
            raise ValueError("Existing partial is larger than the expected download")
        folder = target.with_name(target.name + ".ranges")
        folder.mkdir(exist_ok=True)
        chunk_size = 32 * 1024 * 1024
        ranges = [
            (start, min(start + chunk_size, size) - 1) for start in range(prefix, size, chunk_size)
        ]

        def fetch(bounds):
            start, end = bounds
            path = folder / f"{start:012d}-{end:012d}"
            expected = end - start + 1
            for attempt in range(5):
                have = path.stat().st_size if path.exists() else 0
                if have == expected:
                    return path
                if have > expected:
                    raise ValueError(f"Oversized range file: {path}")
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={start + have}-{end}",
                        "Accept-Encoding": "identity",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content_range = f"bytes {start + have}-{end}/{size}"
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range") != content_range
                        ):
                            raise ValueError("Server returned an unexpected HTTP range")
                        if int(response.headers.get("Content-Length", -1)) != expected - have:
                            raise ValueError("Server returned an unexpected range length")
                        with path.open("ab") as stream:
                            shutil.copyfileobj(response, stream, length=1024 * 1024)
                    if path.stat().st_size != expected:
                        raise OSError("Incomplete range transfer")
                    print(f"Range complete: {start}-{end}", flush=True)
                    return path
                except ValueError:
                    raise
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            raise RuntimeError("Unreachable retry state")

        print(f"Resuming {prefix:,}/{size:,} bytes with {workers} connections", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            paths = list(pool.map(fetch, ranges))
        # Append in byte order only after all ranges are complete. If assembly is
        # interrupted, the next invocation resumes after its valid byte prefix.
        with partial.open("ab") as stream:
            for path in paths:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
        if partial.stat().st_size != size:
            raise ValueError("Assembled size mismatch; preserve files for inspection")
        partial.replace(target)
        print(f"Transport complete; run checksum verification: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.bytes <= 0 or not 1 <= args.workers <= 16:
        parser.error("Expected positive bytes and 1-16 workers")
    download(args.url, args.output, args.bytes, args.workers)
