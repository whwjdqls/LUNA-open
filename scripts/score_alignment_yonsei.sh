#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export PYTHONPATH="$PWD/src"
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
for diagnostic_group in alignment native canvas; do
    "$LUNA_PYTHON" scripts/score_alignment_diagnostics.py --group "$diagnostic_group"
done
"$LUNA_PYTHON" scripts/render_diagnostic_identity_orbit.py
