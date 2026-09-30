"""Download a public, range-capable asset with length checks and a hash receipt."""

import argparse
import os
import socket
import urllib.request
from pathlib import Path

import fetch_lhm_prior_members as ranges


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(urllib.request.Request(args.url, method="HEAD"), timeout=120) as response:
        size = int(response.headers["Content-Length"])
    ranges.URL = args.url
    ranges.download_member(dict(name=args.output.name, size=size, offset=0, regular=True), args.output.parent)


if __name__ == "__main__":
    main()
