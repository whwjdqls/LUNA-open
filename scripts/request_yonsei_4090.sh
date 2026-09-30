#!/usr/bin/env bash
# Launch inside a dedicated tmux window. Pass the successful setup-job dependency.
set -euo pipefail
cd /home/whwjdqls99/LUNA-open
dependency=()
if [[ -n "${1:-}" ]]; then
    dependency=(--dependency="afterok:$1")
fi
placement=()
if [[ -n "${LUNA_EXCLUDE_NODES:-}" ]]; then
    placement=(--exclude="$LUNA_EXCLUDE_NODES")
fi
srun --partition=suma_rtx4090 --qos=base_qos --nodes=1 --ntasks=1 \
    --gres=gpu:RTX4090:1 --cpus-per-task=8 --mem=64G --time=3-00:00:00 \
    --job-name=luna-4090 "${dependency[@]}" "${placement[@]}" --pty \
    bash scripts/yonsei_gpu_session.sh
