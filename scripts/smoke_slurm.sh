#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
"$LUNA_PYTHON" -m pytest -q
flock "$TORCH_EXTENSIONS_DIR/.luna-build.lock" "$LUNA_PYTHON" scripts/smoke_gpu.py \
  --data-root "$LUNA_WORK/data/neuman/dataset" \
  --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
  --assets "$LUNA_WORK/assets" \
  --output "$LUNA_WORK/outputs/smoke-${SLURM_JOB_ID:-local}"
