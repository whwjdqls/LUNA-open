#!/usr/bin/env bash
# Build the pointops extension required by native LHM++ imports, on a CPU allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run native extension compilation inside a Slurm allocation."
  exit 1
fi
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
export MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions_baselines"
baseline_python="$LUNA_WORK/envs/lhm/bin/python"
reference="$LUNA_WORK/references/LHM-plusplus"
[[ "$(git -C "$reference" rev-parse HEAD)" == 906b5d9fb967ab42efb92f6fa55bf22cac86b653 ]]
git -C "$reference" diff --quiet HEAD
"$baseline_python" -m pip check
# Upstream tracks egg-info files. Build from a commit export so package metadata
# generation cannot modify the source checkout used by later model audits.
mkdir -p "$LUNA_WORK/build"
pointops_build="$LUNA_WORK/build/pointops-$SLURM_JOB_ID"
mkdir "$pointops_build"
git -C "$reference" archive HEAD lib/pointops | tar -x -C "$pointops_build"
"$baseline_python" -m pip install --no-build-isolation --no-deps "$pointops_build/lib/pointops"
"$baseline_python" -m pip check
# Import/operator checks are separate so a later dependency fix does not require
# recompiling pointops. Run scripts/audit_lhmpp_dependencies.py after installing
# the supported FlashAttention build.
