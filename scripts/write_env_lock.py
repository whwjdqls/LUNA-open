"""Record installed versions without editable-VCS or local conda build paths."""

import argparse
from importlib.metadata import distributions
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("requirements-resolved.txt"))
args = parser.parse_args()
packages = sorted(
    {
        f"{d.metadata['Name']}=={d.version}"
        for d in distributions()
        if d.metadata["Name"].lower() != "luna-open"
    },
    key=str.lower,
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    "# Verified environment snapshot; reinstall project with pip install --no-deps -e .\n"
    "--extra-index-url https://download.pytorch.org/whl/cu128\n" + "\n".join(packages) + "\n"
)
