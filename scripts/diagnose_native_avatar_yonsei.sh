#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
diagnostic_libraries=""
for diagnostic_lib in "$LUNA_WORK/envs/lhm-native/lib/python3.10/site-packages/nvidia/"*/lib; do
    diagnostic_libraries="$diagnostic_libraries$diagnostic_lib:"
done
export LD_LIBRARY_PATH="$diagnostic_libraries$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
export MAX_JOBS=4
export PYTHONPATH="$PWD/src"
exec 8>"$LUNA_WORK/baselines/lhm-20260928/native-runtime.lock"
flock 8
diagnostic_script=scripts/diagnose_native_avatar.py
if [[ "${1:-}" == "--pose-oracle" ]]; then
    diagnostic_script=scripts/diagnose_lhm_pose_oracle.py
    shift
elif [[ "${1:-}" == "--shared-shape" ]]; then
    diagnostic_script=scripts/fit_shared_reference_shape.py
    shift
fi
exec "$LUNA_WORK/envs/lhmpp-native/bin/python" "$diagnostic_script" "$@"
