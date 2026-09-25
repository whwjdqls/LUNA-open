#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
for kind in body face; do
  "$LUNA_PYTHON" scripts/cache_features.py \
    --data-root "$LUNA_WORK/data/neuman/dataset" \
    --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
    --assets "$LUNA_WORK/assets" \
    --output "$LUNA_WORK/features/neuman-v2" --kind "$kind"
done
