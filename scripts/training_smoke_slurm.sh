#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
config=configs/neuman_smoke.yaml
output="$LUNA_WORK/runs/neuman-smoke-v1"
if [[ -e "$output/identity/latest.pt" || -e "$output/animator/latest.pt" ]]; then
  echo "Existing smoke checkpoints: select a new output path/config before rerunning."
  exit 1
fi
# Hold the extension lock for the sequence to avoid gsplat import/build races.
exec 9>"$TORCH_EXTENSIONS_DIR/.luna-build.lock"
flock 9
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity --stop-after-update 4
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage identity --resume "$output/identity/latest.pt"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator --identity-checkpoint "$output/identity/best.pt" --stop-after-update 4
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator --resume "$output/animator/latest.pt"
"$LUNA_PYTHON" -m luna_open.training --config "$config" --stage animator --resume "$output/animator/best.pt" --evaluate val
