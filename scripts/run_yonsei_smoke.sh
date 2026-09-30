#!/usr/bin/env bash
# Run inside the single-4090 tmux/srun shell after selecting the CUDA compiler.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A Slurm allocation is required." >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export TORCH_CUDA_ARCH_LIST=8.9
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || { echo "Set CUDA_HOME to the compute node's toolkit." >&2; exit 2; }
export PATH="$CUDA_HOME/bin:$PATH"
"$LUNA_PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 1
assert '4090' in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (8, 9)
PY
output="$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID"
mkdir -p "$output"
exec > >(tee -a "$LUNA_WORK/logs/rtx4090-$SLURM_JOB_ID-smoke.log") 2>&1
echo "Starting smoke on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
"$CUDA_HOME/bin/nvcc" --version
"$LUNA_PYTHON" -m ruff check src tests scripts
"$LUNA_PYTHON" -m ruff format --check src tests scripts
for luna_script in scripts/*yonsei*.sh; do bash -n "$luna_script"; done
# gsplat can remove its own JIT lock; serialize build and render access locally.
exec 9>"$TORCH_EXTENSIONS_DIR/.luna-build.lock"
flock 9
"$LUNA_PYTHON" -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("gsplat sm_89 extension imported")'
"$LUNA_PYTHON" scripts/smoke_gpu.py \
    --data-root "$LUNA_WORK/data/neuman/dataset" \
    --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
    --assets "$LUNA_WORK/assets" --output "$output/network-smoke"
for kind in body face motion; do
    echo "Caching $kind features at $(date -Is)"
    "$LUNA_PYTHON" scripts/cache_features.py \
        --data-root "$LUNA_WORK/data/neuman/dataset" \
        --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
        --assets "$LUNA_WORK/assets" --output "$LUNA_WORK/features/neuman-v2" \
        --kind "$kind"
done
"$LUNA_PYTHON" scripts/verify_features.py \
    --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
    --features "$LUNA_WORK/features/neuman-v2" --kinds body face motion \
    --output "$output/features-audit.json"
config=configs/neuman_yonsei_smoke.yaml
run="$LUNA_WORK/runs/neuman-smoke-4090-v1"
if [[ -e "$run/identity/train.jsonl" || -e "$run/animator/train.jsonl" ]]; then
    echo "Existing smoke run at $run; inspect and explicitly resume it, or select a new config." >&2
    exit 2
fi
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity --stop-after-update 4
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity --resume "$run/identity/latest.pt"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator \
    --identity-checkpoint "$run/identity/best.pt" --stop-after-update 4
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator --resume "$run/animator/latest.pt"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity \
    --resume "$run/identity/best.pt" --evaluate val
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator \
    --resume "$run/animator/best.pt" --evaluate val
"$LUNA_PYTHON" scripts/audit_training_run.py --config "$config" \
    --output "$output/training-smoke-audit.json"
echo "Yonsei network, feature and two-stage training smoke completed at $(date -Is)"
