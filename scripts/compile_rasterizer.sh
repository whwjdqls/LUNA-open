#!/usr/bin/env bash
# Run in a CPU or GPU Slurm allocation, not on the login node.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
# gsplat 1.5.3 can remove its own build lock; serialize project build/run access.
flock "$TORCH_EXTENSIONS_DIR/.luna-build.lock" \
  "$LUNA_PYTHON" -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("gsplat sm_100 extension imported")'
