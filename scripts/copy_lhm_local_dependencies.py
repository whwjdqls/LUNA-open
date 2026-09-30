"""Copy compatible Python 3.10 dependencies without editing the source environment."""

import importlib.metadata as metadata
import json
import os
import shutil
import socket
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    root = Path("/scratch2/whwjdqls99/LUNA-open")
    source = Path("/home/whwjdqls99/miniconda3/envs/robocasa/lib/python3.10/site-packages")
    target = root / "envs/lhm-native/lib/python3.10/site-packages"
    index = {
        canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions(path=[str(source)])
    }
    installed = {
        canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions(path=[str(target)])
    }
    queue = [
        Requirement(item)
        for item in (
            "transformers==4.41.2",
            "opencv-python==4.11.0.86",
            "numba==0.59.1",
            "scipy==1.13.1",
            "filelock",
            "typing_extensions",
            "networkx",
            "jinja2",
            "sympy",
            "fsspec",
            "mpmath",
            "safetensors",
            "loguru",
            "imageio",
            "imageio-ffmpeg",
            "matplotlib",
            "PyYAML",
            "tqdm",
            "requests",
            "psutil",
            "packaging",
            "tokenizers==0.19.1",
            "numpy==1.23.5",
        )
    ]
    environment = dict(
        default_environment(), python_version="3.10", python_full_version="3.10.19", extra=""
    )
    copied, unavailable, visited = [], [], set()
    while queue:
        req = queue.pop(0)
        name = canonicalize_name(req.name)
        if name in visited or (req.marker and not req.marker.evaluate(environment)):
            continue
        visited.add(name)
        if name in installed and installed[name].version in req.specifier:
            continue
        dist = index.get(name)
        if dist is None or dist.version not in req.specifier:
            unavailable.append(str(req))
            continue
        roots = {Path(str(file)).parts[0] for file in (dist.files or [])}
        for part in sorted(roots):
            if part in ("..", "."):
                continue
            item = source / part
            if not item.exists():
                continue
            destination = target / part
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
        copied.append(dict(name=name, version=dist.version, roots=sorted(roots)))
        queue.extend(Requirement(item) for item in (dist.requires or []))
        print(f"Copied {name} {dist.version}", flush=True)
    (root / "baselines/lhm-20260928/copied-local-dependencies.json").write_text(
        json.dumps(dict(source=str(source), copied=copied, unavailable=unavailable), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
