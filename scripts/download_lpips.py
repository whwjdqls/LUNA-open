"""Prefetch LPIPS's public AlexNet backbone with curl in a CPU allocation."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from luna_open.perceptual import (
    ALEXNET_BYTES,
    ALEXNET_FILENAME,
    ALEXNET_SHA256,
    ALEXNET_URL,
    verify_alexnet,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, default=os.environ.get("TORCH_HOME"))
    args = parser.parse_args()
    if args.torch_home is None:
        parser.error("Source scripts/parcc_env.sh or provide --torch-home in project storage")
    folder = args.torch_home / "hub/checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ALEXNET_FILENAME
    if not target.exists():
        partial = target.with_name(target.name + ".luna-part")
        if not partial.exists() or partial.stat().st_size != ALEXNET_BYTES:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "600",
                    "--continue-at",
                    "-",
                    "--silent",
                    "--show-error",
                    "--output",
                    str(partial),
                    ALEXNET_URL,
                ],
                check=True,
            )
        verify_alexnet(partial)
        partial.replace(target)
    verify_alexnet(target)
    report = dict(
        source=ALEXNET_URL,
        path=str(target),
        bytes=ALEXNET_BYTES,
        sha256=ALEXNET_SHA256,
        source_of_pin="torchvision filename prefix; full SHA256 verified locally from official URL",
    )
    receipt = target.with_name(target.name + ".receipt.json")
    temporary = receipt.with_suffix(".part")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(receipt)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
