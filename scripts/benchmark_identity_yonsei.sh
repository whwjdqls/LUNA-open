#!/usr/bin/env bash
# Run inside the separate tmux/srun GPU shell authorized for batch benchmarks.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "${SLURMD_NODENAME:-}" != *login* ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export TORCH_CUDA_ARCH_LIST=8.9
export PATH="$CUDA_HOME/bin:$PATH"
bench="$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/identity-batching"
[[ ! -e "$bench" ]] || { echo "Benchmark directory already exists"; exit 2; }
mkdir -p "$bench/source" "$bench/provenance"
exec > >(tee -a "$bench/launch.log") 2>&1
"$LUNA_PYTHON" -m ruff format src/luna_open/identity_batching.py scripts/benchmark_identity_batching.py
"$LUNA_PYTHON" -m ruff check src/luna_open/identity_batching.py scripts/benchmark_identity_batching.py
cp -a src scripts configs pyproject.toml requirements-resolved.txt "$bench/source/"
git diff --binary > "$bench/provenance/working-tree.diff"
git status --short > "$bench/provenance/git-status.txt"
git rev-parse HEAD > "$bench/provenance/git-head.txt"
cp "$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID/allocation.json" "$bench/provenance/allocation.json"
cd "$bench/source"
export PYTHONPATH="$PWD/src"
"$LUNA_PYTHON" scripts/benchmark_identity_batching.py \
 --checkpoint "$LUNA_WORK/runs/neuman-identity-v2-20260928/identity/latest.pt" \
 --output "$bench/results" --warmup 3 --steps 10 --rounds 2
