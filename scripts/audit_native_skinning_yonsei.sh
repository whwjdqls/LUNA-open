#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
audit_libs=""
for audit_lib in "$LUNA_WORK/envs/lhm-native/lib/python3.10/site-packages/nvidia/"*/lib; do
    audit_libs="$audit_libs$audit_lib:"
done
export LD_LIBRARY_PATH="$audit_libs$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
export PYTHONPATH="$PWD/src"
exec 8>"$LUNA_WORK/baselines/lhm-20260928/native-runtime.lock"
flock 8
audit_script=scripts/audit_native_skinning_buffers.py
if [[ "${1:-}" == "--learned" ]]; then
    audit_script=scripts/audit_learned_gaussian_geometry.py
    shift
fi
exec "$LUNA_WORK/envs/lhmpp-native/bin/python" "$audit_script" "$@"
