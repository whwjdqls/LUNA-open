#!/usr/bin/env bash
# Run both native trainability gates sequentially within one B200 allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'GPU allocation required' >&2; exit 2; }
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions_baselines"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
native_smoke_status=0
for method in lhmpp lhm; do
    if [[ "$method" == lhm ]]; then
        native_source="$LUNA_WORK/references/LHM"
        architecture="$LUNA_WORK/assets/lhm_500m/config.json"
    else
        native_source="$LUNA_WORK/references/LHM-plusplus"
        architecture="$LUNA_WORK/assets/lhmpp_700m/config.json"
    fi
    if "$LUNA_WORK/envs/lhm/bin/python" scripts/smoke_native_training.py \
        --method "$method" --source "$native_source" \
        --architecture "$architecture" --runtime "$LUNA_WORK/baselines/$method-runtime-v1" \
        --training-source "$LUNA_WORK/references/LHM-plusplus" \
        --data-root "$LUNA_WORK/data/neuman/dataset" \
        --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
        --output "$LUNA_WORK/outputs/native-training-$SLURM_JOB_ID/$method" \
        > "$LUNA_WORK/logs/native-training-$method-$SLURM_JOB_ID.log" 2>&1; then
        echo "$method: passed"
    else
        native_smoke_status=1
        echo "$method: failed; inspect native-training-$method-$SLURM_JOB_ID.log" >&2
    fi
done
exit "$native_smoke_status"
