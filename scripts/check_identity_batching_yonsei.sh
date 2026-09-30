#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "${SLURMD_NODENAME:-}" != *login* ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export TORCH_CUDA_ARCH_LIST=8.9
export PATH="$CUDA_HOME/bin:$PATH"
bench="$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/identity-batching"
exec > >(tee -a "$bench/checks.log") 2>&1
"$LUNA_PYTHON" -m ruff check --fix scripts/check_identity_batch_loss.py scripts/profile_identity_batching.py
"$LUNA_PYTHON" -m ruff format scripts/check_identity_batch_loss.py scripts/profile_identity_batching.py
if [[ -f "$bench/source/scripts/check_identity_batch_loss.py" ]]; then
 cp -n "$bench/source/scripts/check_identity_batch_loss.py" "$bench/source/scripts/check_identity_batch_loss_v1.py"
fi
cp scripts/check_identity_batch_loss.py scripts/profile_identity_batching.py "$bench/source/scripts/"
cd "$bench/source"
export PYTHONPATH="$PWD/src"
if "$LUNA_PYTHON" scripts/check_identity_batch_loss.py --output "$bench/results/loss-check-default.json"; then
 echo "Default-precision loss check passed."
else
 echo "Default-precision loss check did not pass; details retained in log and report."
fi
"$LUNA_PYTHON" scripts/check_identity_batch_loss.py --disable-tf32 --output "$bench/results/loss-check-fp32.json"
"$LUNA_PYTHON" scripts/profile_identity_batching.py --benchmark "$bench/results"
