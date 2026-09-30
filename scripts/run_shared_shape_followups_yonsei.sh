#!/usr/bin/env bash
# Wider-bound sensitivity and common scoring, inside the existing 4090 allocation.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export PYTHONPATH="$PWD/src"
"$LUNA_PYTHON" - <<'PY'
import json
from pathlib import Path
root=Path('/scratch2/whwjdqls99/LUNA-open/diagnostics/shared-shape-20260929')
for method in ('lhm','lhmpp'):
    assert json.loads((root/method/'audit.json').read_text())['complete']
PY
for shape_method in lhm lhmpp; do
    bash scripts/diagnose_native_avatar_yonsei.sh --shared-shape \
        --method "$shape_method" --max-delta 10 \
        --output "$LUNA_WORK/diagnostics/shared-shape-20260929/$shape_method-wide" \
        > "$LUNA_WORK/logs/shared-shape-$shape_method-wide-20260929.log" 2>&1
done
"$LUNA_PYTHON" scripts/score_alignment_diagnostics.py --group shape \
    > "$LUNA_WORK/logs/shared-shape-metrics-20260929.log" 2>&1
echo "Shared-shape sensitivity and scoring complete"
