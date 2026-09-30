#!/usr/bin/env bash
# Run from the new tmux/srun compute-node shell. Preserve both old experiments
# and an immutable copy of the source used for this fresh identity run.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "${SLURMD_NODENAME:-}" != *login* ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export TORCH_CUDA_ARCH_LIST=8.9
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/src"
run="$LUNA_WORK/runs/neuman-identity-v2-20260928"
[[ ! -e "$run" ]] || { echo "Run directory exists; inspect its checkpoint and resume explicitly."; exit 2; }
mkdir -p "$run/provenance" "$run/source"
exec > >(tee -a "$run/launch.log") 2>&1
echo "Identity v2 starts on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.free --format=csv
"$LUNA_PYTHON" -m ruff format src/luna_open/identity_training.py scripts/audit_identity_retraining.py
"$LUNA_PYTHON" -m ruff check src/luna_open/model.py src/luna_open/features.py \
 src/luna_open/pipeline.py src/luna_open/identity_training.py scripts/cache_features.py \
 scripts/diagnose_identity.py scripts/audit_identity_retraining.py
git diff --binary > "$run/provenance/working-tree.diff"
git status --short > "$run/provenance/git-status.txt"
git rev-parse HEAD > "$run/provenance/git-head.txt"
cp -a src scripts configs pyproject.toml requirements-resolved.txt "$run/source/"
cp "$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/allocation.json" "$run/provenance/allocation.json"
cd "$run/source"
export PYTHONPATH="$PWD/src"
config=configs/neuman_yonsei_identity_v2.yaml
"$LUNA_PYTHON" scripts/audit_identity_retraining.py --config "$config" \
 --output "$run/provenance/preflight.json" \
 --legacy-checkpoint "$LUNA_WORK/outputs/gpu-2336972/identity-10000-train-test/identity-snapshot.pt"
# The first two real updates use the full 20k schedule. A fresh-process resume
# then continues the same run after validating checkpoint contents and inputs.
"$LUNA_PYTHON" -m luna_open.identity_training --config "$config" --stop-after-update 2
"$LUNA_PYTHON" scripts/audit_identity_retraining.py --config "$config" \
 --checkpoint "$run/identity/latest.pt" --output "$run/provenance/update-2-audit.json"
"$LUNA_PYTHON" -m luna_open.identity_training --config "$config" --resume "$run/identity/latest.pt"
"$LUNA_PYTHON" scripts/audit_identity_retraining.py --config "$config" \
 --checkpoint "$run/identity/latest.pt" --output "$run/provenance/final-checkpoint-audit.json"
echo "Identity v2 training finished at $(date -Is)"
