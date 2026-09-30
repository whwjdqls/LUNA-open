#!/usr/bin/env bash
# Source from the repository root on Yonsei. No global shell settings are edited.
export LUNA_WORK="${LUNA_WORK:-/scratch2/whwjdqls99/LUNA-open}"
# A dedicated training environment is separate from the acquisition environment.
export LUNA_PYTHON="${LUNA_PYTHON:-$LUNA_WORK/envs/luna/bin/python}"
export PATH="$(dirname "$LUNA_PYTHON"):$PATH"
export LUNA_DATA_PYTHON="${LUNA_DATA_PYTHON:-$LUNA_WORK/envs/data-tools/bin/python}"
export LUNA_SMPL_PYTHON="${LUNA_SMPL_PYTHON:-$LUNA_WORK/envs/smpl-convert/bin/python}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_HOME="$LUNA_WORK/cache/huggingface"
export PIP_CACHE_DIR="$LUNA_WORK/cache/pip"
export TORCH_HOME="$LUNA_WORK/cache/torch"
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions"
export XDG_CACHE_HOME="$LUNA_WORK/cache"
export RUFF_CACHE_DIR="$LUNA_WORK/cache/ruff"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }-o cache_dir=$LUNA_WORK/cache/pytest"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
    "$RUFF_CACHE_DIR" "$LUNA_WORK/cache/pytest" "$LUNA_WORK/logs"
