#!/usr/bin/env bash
# Invoke within one allocated GPU shell: SERVER METHOD --output ... [trainer flags].
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && $# -ge 2 ]] || {
    echo 'Usage in one GPU allocation: train_comparison.sh parcc|yonsei lhm|lhmpp|luna --output DIR [flags]' >&2
    exit 2
}
comparison_server="$1"
comparison_method="$2"
shift 2
cd "$(dirname "$0")/.."
case "$comparison_server" in
    parcc)
        source scripts/parcc_env.sh
        source /usr/share/lmod/lmod/init/bash
        module load cuda/12.8.1
        comparison_lhm_python="${LUNA_COMPARE_LHM_PYTHON:-$LUNA_WORK/envs/lhm/bin/python}"
        comparison_lhmpp_python="${LUNA_COMPARE_LHMPP_PYTHON:-$LUNA_WORK/envs/lhm/bin/python}"
        ;;
    yonsei)
        source scripts/yonsei_env.sh
        comparison_lhm_python="${LUNA_COMPARE_LHM_PYTHON:-$LUNA_WORK/envs/lhm-native/bin/python}"
        comparison_lhmpp_python="${LUNA_COMPARE_LHMPP_PYTHON:-$LUNA_WORK/envs/lhmpp-native/bin/python}"
        if [[ "$comparison_method" == luna ]]; then
            export CUDA_HOME="${LUNA_COMPARE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/12.8}"
        else
            export CUDA_HOME="${LUNA_COMPARE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/11.8}"
        fi
        export PATH="$CUDA_HOME/bin:$PATH"
        ;;
    *) echo 'Unknown server' >&2; exit 2 ;;
esac
case "$comparison_method" in
    luna) comparison_python="$LUNA_PYTHON" ;;
    lhm) comparison_python="$comparison_lhm_python" ;;
    lhmpp) comparison_python="$comparison_lhmpp_python" ;;
    *) echo 'Unknown method' >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST="$($comparison_python -c 'import torch; assert torch.cuda.device_count() == 1; print("%d.%d" % torch.cuda.get_device_capability())')"
# Environment and device separate binary extension caches on both servers.
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/comparison-extensions-$comparison_method-sm${TORCH_CUDA_ARCH_LIST/./}"
exec "$comparison_python" -m luna_open.comparison_training \
    --method "$comparison_method" --contract configs/comparison.yaml \
    --corpus "configs/comparison_neuman_$comparison_server.yaml" "$@"
