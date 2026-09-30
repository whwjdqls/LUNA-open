#!/usr/bin/env bash
# Run through Slurm on a compute node. Keep all environments in scratch.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A Slurm allocation is required." >&2; exit 2; }
cd "${LUNA_REPO:-${SLURM_SUBMIT_DIR:-/home/whwjdqls99/LUNA-open}}"
source scripts/yonsei_env.sh
echo "Setting up LUNA on $(hostname), Slurm job $SLURM_JOB_ID"
exec 9>"$LUNA_WORK/envs/.luna-setup.lock"
flock --nonblock 9 || { echo "Another environment setup is active." >&2; exit 2; }
if [[ ! -f "$LUNA_WORK/envs/luna/pyvenv.cfg" ]]; then
    "$LUNA_DATA_PYTHON" -m venv "$LUNA_WORK/envs/luna"
fi
export PATH="$LUNA_WORK/envs/luna/bin:$PATH"
"$LUNA_PYTHON" -m pip install --disable-pip-version-check -r requirements-resolved.txt
"$LUNA_PYTHON" -m pip install --disable-pip-version-check --no-deps -e .
"$LUNA_PYTHON" -m pip check
"$LUNA_PYTHON" scripts/write_env_lock.py --output "$LUNA_WORK/envs/luna/requirements-resolved.txt"
"$LUNA_PYTHON" -m pytest -q
"$LUNA_PYTHON" scripts/smoke_metrics_cpu.py \
    --root "$LUNA_WORK/data/neuman/dataset" \
    --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
    --output "$LUNA_WORK/outputs/yonsei-lpips-cpu-smoke.json"
"$LUNA_PYTHON" - <<'PY'
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

root = Path(os.environ['LUNA_WORK'])
receipt = dict(
    completed_at=datetime.now(timezone.utc).isoformat(),
    slurm_job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
    python=platform.python_version(), interpreter=os.environ['LUNA_PYTHON'],
    requirements_sha256=hashlib.sha256(Path('requirements-resolved.txt').read_bytes()).hexdigest(),
    versions={name: version(name) for name in
              ('torch', 'torchvision', 'numpy', 'gsplat', 'transformers', 'smplx', 'lpips')},
    pip_check='passed', cpu_tests='passed', lpips_cpu_smoke='passed',
)
path = root / 'envs/luna/setup-receipt.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
PY
