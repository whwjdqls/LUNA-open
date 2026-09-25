"""Fetch pinned upstream baseline repositories into external project storage."""

import argparse
import subprocess
from pathlib import Path

REPOS = {
    "LHM": ("https://github.com/aigc3d/LHM.git", "4f88aaeb3629249fbbddb4d0784a06962d9e1338"),
    "LHM-plusplus": (
        "https://github.com/aigc3d/LHM-plusplus.git",
        "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in REPOS.items():
        path = args.root / name
        if path.exists():
            current = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if current != revision:
                raise ValueError(
                    f"Existing checkout differs; preserve it and choose another root: {path}"
                )
            continue
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(path)], check=True
        )
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        print(name, revision, flush=True)


if __name__ == "__main__":
    main()
