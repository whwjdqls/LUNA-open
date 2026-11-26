#!/usr/bin/env bash
# Run inside a one-GPU B200 Slurm step.
set -euo pipefail
mode=${1:?expected smoke, pilot, wan-pilot, or train}
arm=${2:?expected A, B, C, or D}
output=${3:?expected a new experiment root}
shift 3
repo=$(cd "$(dirname "$0")/.." && pwd)
work=/vast/projects/lingjie6/impossible/jungbinc
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" TORCH_HOME="$work/cache/torch"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python="$work/envs/luna/bin/python"
options=(--geometry-monitor)
case "$arm" in
    A) ;;
    B) options+=(--canonical-regularization) ;;
    C) options+=(--coarse-to-fine) ;;
    D) options+=(--coarse-to-fine --canonical-regularization) ;;
    *) exit 2 ;;
esac
case "$mode" in
    pilot) module=fit_native_tokens ;;
    smoke|wan-pilot) module=verify_native_uv
           options+=(--legacy-anchor-receipt /ceph/projects/lingjie6/impossible/jungbinc/experiments/wan-uv-native-20261125-v1/legacy-anchors.json --smoke-steps 2)
           if [[ $mode == wan-pilot ]]; then options+=(--matched-updates --smoke-steps 50); fi ;;
    train) module=train_wan_mvh ;;
    *) exit 2 ;;
esac
exec "$python" -m "uv_gaussian_avatar.$module" --config "$repo/configs/lunav2_identity.yaml" \
  --wan-source "$work/references/Wan2.2-20261028" --wan-checkpoint "$work/assets/wan22-ti2v-5b-20261028" \
  --wan-deps "$work/envs/wan22-20261028/lib/python3.11/site-packages" \
  --cuda-extension "$work/cache/comparison-extensions-lhm-sm100/gsplat_cuda" \
  --uv-representation native-islands --uv-patch-size 16 --image-side 16 --atlas-size 528 --adaptive-body-scale 1.2 \
  --output "$output/$mode-$arm" --entity jungbinc-upenn "${options[@]}" "$@"
