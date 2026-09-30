#!/usr/bin/env bash
# Run the released baseline construction/operator gates sequentially on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
if [[ -z "${SLURM_JOB_ID:-}" || $# -ne 1 ]]; then
  echo "Usage inside one GPU allocation: $0 /path/to/completed-attention-diagnostic.json"
  exit 1
fi
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions_baselines"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
baseline_python="$LUNA_WORK/envs/lhm/bin/python"
# The diagnostic deliberately fails if the original dispatch is wrong. Require
# its independently checked compatible path before enabling the process setting.
"$baseline_python" - "$1" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["capability"] == [10, 0], "Expected a completed B200 diagnostic"
for package in ("torch", "xformers", "flash-attn"):
    assert version(package) == data["packages"][package], "Dependency changed after diagnosis"
for dtype in ("torch.bfloat16", "torch.float16"):
    for backend in ("torch_math", "flash_attn_direct", "xformers_flash", "xformers_without_flash3"):
        assert data["cases"][dtype][backend]["matches_reference"], (dtype, backend)
    assert data["default_dispatch"][dtype].startswith("fa3"), "Unexpected original backend"
print(json.dumps(dict(
    attention_diagnostic=str(path.resolve()),
    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    compatible_dispatch_verified=True,
)), flush=True)
PY
"$baseline_python" scripts/smoke_lhm.py --method lhm --disable-xformers-flash3 \
  --reference "$LUNA_WORK/references/LHM" --checkpoint "$LUNA_WORK/assets/lhm_500m" \
  --runtime "$LUNA_WORK/baselines/lhm-runtime-v1" \
  --output "$LUNA_WORK/outputs/baseline-smoke-$SLURM_JOB_ID/lhm" \
  > "$LUNA_WORK/logs/lhm-smoke-$SLURM_JOB_ID.log" 2>&1
"$baseline_python" scripts/smoke_lhm.py --method lhmpp --disable-xformers-flash3 \
  --reference "$LUNA_WORK/references/LHM-plusplus" --checkpoint "$LUNA_WORK/assets/lhmpp_700m" \
  --runtime "$LUNA_WORK/baselines/lhmpp-runtime-v1" \
  --output "$LUNA_WORK/outputs/baseline-smoke-$SLURM_JOB_ID/lhmpp" \
  > "$LUNA_WORK/logs/lhmpp-smoke-$SLURM_JOB_ID.log" 2>&1
