#!/usr/bin/env bash
# sbatch --partition=dell_cpu --qos=cpu_qos -N 1 -n 1 -c 4 --mem=8G \
#   --time=02:00:00 --output=/scratch2/whwjdqls99/LUNA-open/logs/acquire-%j.log \
#   scripts/acquire_yonsei_slurm.sh
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script through Slurm; downloads must run on a compute node." >&2
    exit 2
fi
cd "${LUNA_REPO:-${SLURM_SUBMIT_DIR:-/home/whwjdqls99/LUNA-open}}"
source scripts/yonsei_env.sh
exec 9>"$LUNA_WORK/acquisition.lock"
if ! flock --nonblock 9; then
    echo "Another acquisition job owns $LUNA_WORK/acquisition.lock" >&2
    exit 2
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_PROGRESS_BARS=1
job_log="$LUNA_WORK/logs/acquire-$SLURM_JOB_ID"
echo "Yonsei acquisition: job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown} root=$LUNA_WORK"

(
    "$LUNA_DATA_PYTHON" scripts/download_ranges.py \
        --url https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip \
        --output "$LUNA_WORK/data/neuman/dataset.zip" --bytes 2154631653 --workers 8 &&
    "$LUNA_DATA_PYTHON" scripts/acquire_neuman.py --root "$LUNA_WORK/data/neuman" &&
    "$LUNA_DATA_PYTHON" -m luna_open.data.neuman \
        --root "$LUNA_WORK/data/neuman/dataset" \
        --manifest "$LUNA_WORK/data/neuman/manifest-v2.json"
) > "$job_log-neuman.log" 2>&1 &
neuman_pid=$!

(
    "$LUNA_DATA_PYTHON" scripts/download_assets.py --root "$LUNA_WORK/assets" \
        sapiens_body dino_face dino_motion lhm_500m lhmpp_700m &&
    "$LUNA_DATA_PYTHON" scripts/audit_downloaded_assets.py --root "$LUNA_WORK/assets" \
        --report "$LUNA_WORK/assets/download-verification.json"
) > "$job_log-models.log" 2>&1 &
models_pid=$!

(
    "$LUNA_DATA_PYTHON" scripts/download_lpips.py --torch-home "$TORCH_HOME" &&
    if [[ ! -f "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl" ]]; then
        "$LUNA_SMPL_PYTHON" scripts/prepare_smpl.py \
            --source "$PWD/assets/SMPL_NEUTRAL.pkl" \
            --output "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl"
    fi
) > "$job_log-body-lpips.log" 2>&1 &
body_pid=$!

result=0
if wait "$body_pid"; then echo "SMPL and LPIPS prepared"; else result=1; fi
if wait "$models_pid"; then echo "All pinned checkpoint hashes verified"; else result=1; fi
if wait "$neuman_pid"; then echo "NeuMan archive and frame manifest verified"; else result=1; fi
echo "Acquisition result=$result; detailed logs: $job_log-*.log"
exit "$result"
