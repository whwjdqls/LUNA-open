"""Record installed versions without editable-VCS or local conda build paths."""

from importlib.metadata import distributions
from pathlib import Path

packages = sorted(
    {
        f"{d.metadata['Name']}=={d.version}"
        for d in distributions()
        if d.metadata["Name"].lower() != "luna-open"
    },
    key=str.lower,
)
Path("requirements-resolved.txt").write_text(
    "# Verified environment snapshot; reinstall project with pip install --no-deps -e .\n"
    "--extra-index-url https://download.pytorch.org/whl/cu128\n" + "\n".join(packages) + "\n"
)
