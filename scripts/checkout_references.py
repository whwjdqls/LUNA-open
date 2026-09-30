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
    "pytorch3d-baseline": (
        "https://github.com/facebookresearch/pytorch3d.git",
        "978cd99221b9e0a6a568f1d427854d73363265cf",
    ),
    "basicsr-baseline": (
        "https://github.com/XPixelGroup/BasicSR.git",
        "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a",
    ),
    "diff-gaussian-baseline": (
        "https://github.com/ashawkey/diff-gaussian-rasterization.git",
        "8829d14f814fccdaf840b7b0f3021a616583c0a1",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repos", nargs="+", choices=REPOS, default=["LHM", "LHM-plusplus"])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.repos:
        url, revision = REPOS[name]
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
        if (path / ".gitmodules").is_file():
            subprocess.run(
                ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"],
                check=True,
            )
        print(name, revision, flush=True)


if __name__ == "__main__":
    main()
