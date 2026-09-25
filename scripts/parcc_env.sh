#!/usr/bin/env bash
# Source in a job shell. No global shell configuration is modified.
export LUNA_WORK="${LUNA_WORK:-/vast/projects/lingjie6/impossible/jungbinc}"
export LUNA_PYTHON="${LUNA_PYTHON:-$LUNA_WORK/envs/luna/bin/python}"
export PATH="$(dirname "$LUNA_PYTHON"):$PATH"
export HF_HOME="$LUNA_WORK/cache/huggingface"
export PIP_CACHE_DIR="$LUNA_WORK/cache/pip"
export TORCH_HOME="$LUNA_WORK/cache/torch"
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions"
export XDG_CACHE_HOME="$LUNA_WORK/cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$LUNA_WORK/logs"
