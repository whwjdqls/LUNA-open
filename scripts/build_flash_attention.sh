#!/usr/bin/env bash
# Build the supported external FlashAttention backend without changing the env.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run CUDA extension compilation inside a Slurm allocation."
  exit 1
fi
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=100
export MAX_JOBS=4
export NVCC_THREADS=2
baseline_python="$LUNA_WORK/envs/lhm/bin/python"
source_archive="$LUNA_WORK/sources/flash_attn-2.8.2.tar.gz"
source_sha256=740a5370f406cbe16155cc6d078ec543a97a137501f32cba89198295e6b80e54
printf '%s  %s\n' "$source_sha256" "$source_archive" | sha256sum --check
build_dir="$LUNA_WORK/build/flash-attn-$SLURM_JOB_ID"
mkdir "$build_dir"
export TMPDIR="$build_dir/tmp"
mkdir "$TMPDIR"
"$baseline_python" -m pip check
"$baseline_python" -m pip wheel --verbose --no-build-isolation --no-deps \
  --wheel-dir "$build_dir" "$source_archive"
"$baseline_python" - "$build_dir" "$source_archive" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch

build_dir, source = map(Path, sys.argv[1:])
wheels = list(build_dir.glob("flash_attn-2.8.2-*.whl"))
if len(wheels) != 1:
    raise ValueError(f"Expected one built FlashAttention 2.8.2 wheel, found {wheels}")
wheel = wheels[0]


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


report = dict(
    job_id=os.environ["SLURM_JOB_ID"],
    source=fingerprint(source),
    wheel=fingerprint(wheel),
    torch=version("torch"),
    cuda=torch.version.cuda,
    cxx11_abi=torch.compiled_with_cxx11_abi(),
    compiler=subprocess.check_output(["nvcc", "--version"], text=True),
    build_flags={key: os.environ[key] for key in (
        "FLASH_ATTENTION_FORCE_BUILD", "FLASH_ATTN_CUDA_ARCHS", "MAX_JOBS", "NVCC_THREADS"
    )},
    limitation="Wheel compilation only; not installed or executed on GPU",
)
(build_dir / "build.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
