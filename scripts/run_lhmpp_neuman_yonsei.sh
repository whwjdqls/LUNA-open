#!/usr/bin/env bash
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm compute node required' >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
lhmpp_libraries=""
for lhmpp_lib in "$LUNA_WORK/envs/lhm-native/lib/python3.10/site-packages/nvidia/"*/lib; do
    lhmpp_libraries="$lhmpp_libraries$lhmpp_lib:"
done
export LD_LIBRARY_PATH="$lhmpp_libraries$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
export MAX_JOBS=4
export PYTHONPATH="$PWD/src"
lhmpp_base="$LUNA_WORK/baselines/lhmpp-20260928"
lhm_base="$LUNA_WORK/baselines/lhm-20260928"
exec 8>"$lhm_base/native-runtime.lock"
flock 8
exec "$LUNA_WORK/envs/lhmpp-native/bin/python" scripts/evaluate_lhmpp_neuman.py \
    --source "$lhmpp_base/source" --checkpoint "$LUNA_WORK/assets/lhmpp_700m" \
    --protocol "$lhm_base/protocol-test" --fits "$lhm_base/fit-test/fits.json" \
    --canonical-camera "$lhm_base/ours-identity-14750-test/canonical-camera.json" "$@"
