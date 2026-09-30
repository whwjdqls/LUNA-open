#!/usr/bin/env bash
# Start a fresh full NeuMan development run after inspecting the smoke evidence.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A Slurm allocation is required." >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export TORCH_CUDA_ARCH_LIST=8.9
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || { echo "Set CUDA_HOME to the compute node's toolkit." >&2; exit 2; }
export PATH="$CUDA_HOME/bin:$PATH"
export LUNA_SMOKE_REPORT="${LUNA_SMOKE_REPORT:-$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/training-smoke-audit.json}"
"$LUNA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import torch

assert torch.cuda.device_count() == 1
assert '4090' in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (8, 9)
report = json.loads(Path(os.environ['LUNA_SMOKE_REPORT']).read_text())
assert all(report['stages'][stage]['completed_updates'] == 8 for stage in ('identity', 'animator'))
PY
run="$LUNA_WORK/runs/neuman"
if [[ -e "$run/identity/train.jsonl" || -e "$run/animator/train.jsonl" ]]; then
    echo "Existing run at $run: inspect and use the CLI's --resume option." >&2
    exit 2
fi
mkdir -p "$run/provenance"
exec > >(tee -a "$LUNA_WORK/logs/rtx4090-$SLURM_JOB_ID-training.log") 2>&1
echo "NeuMan training starts on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
cp "$LUNA_SMOKE_REPORT" "$run/provenance/training-smoke-audit.json"
cp "$LUNA_WORK/envs/luna/setup-receipt.json" "$run/provenance/environment.json"
cp "$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/allocation.json" "$run/provenance/allocation.json"
git rev-parse HEAD > "$run/provenance/git-head.txt"
git diff --binary > "$run/provenance/working-tree.diff"
tar -czf "$run/provenance/source-at-start.tar.gz" \
    --exclude='__pycache__' src scripts configs tests pyproject.toml requirements-resolved.txt
"$LUNA_PYTHON" -m ruff check src tests scripts
"$LUNA_PYTHON" -m ruff format --check src tests scripts
"$LUNA_PYTHON" -m pytest -q
exec 9>"$TORCH_EXTENSIONS_DIR/.luna-build.lock"
flock 9
config=configs/neuman_yonsei.yaml
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity
echo "Identity stage finished at $(date -Is)"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity \
    --resume "$run/identity/best.pt" --evaluate val
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity \
    --resume "$run/identity/best.pt" --evaluate test
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator \
    --identity-checkpoint "$run/identity/best.pt"
echo "Animator stage finished at $(date -Is)"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator \
    --resume "$run/animator/best.pt" --evaluate val
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator \
    --resume "$run/animator/best.pt" --evaluate test
"$LUNA_PYTHON" scripts/audit_training_run.py --config "$config" --output "$run/completion-audit.json"
echo "NeuMan development training and held-out evaluation completed at $(date -Is)"
