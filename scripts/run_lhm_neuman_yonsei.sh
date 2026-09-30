#!/usr/bin/env bash
# Run inside the allocated one-4090 tmux shell.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
lhm_library_path=""
for lhm_library in "$LUNA_WORK/envs/lhm-native/lib/python3.10/site-packages/nvidia/"*/lib; do
    lhm_library_path="$lhm_library_path$lhm_library:"
done
# Avoid mixing system cuDNN with the cuDNN bundled alongside Torch 2.3.
export LD_LIBRARY_PATH="$lhm_library_path$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
export MAX_JOBS=4
export PYTHONPATH="$PWD/src"
lhm_base="$LUNA_WORK/baselines/lhm-20260928"
lhm_identity_export="${LHM_IDENTITY_EXPORT:-$lhm_base/ours-identity-14750-test}"
# gsplat 1.4 removes its JIT lock/build directory on a first import. Serialize
# native invocations so simultaneous initial imports cannot destroy a build.
exec 8>"$lhm_base/native-runtime.lock"
flock 8
exec "$LUNA_WORK/envs/lhm-native/bin/python" scripts/evaluate_lhm_neuman.py \
    --source "$lhm_base/source" --checkpoint "$LUNA_WORK/assets/lhm_500m" \
    --protocol "$lhm_base/protocol-test" \
    --face-inputs "$lhm_identity_export" \
    --fits "$lhm_base/fit-test/fits.json" \
    --canonical-camera "$lhm_identity_export/canonical-camera.json" "$@"
