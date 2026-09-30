#!/usr/bin/env bash
# Entrypoint for the explicitly requested tmux + srun single-4090 session.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A Slurm GPU allocation is required." >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export PYTHONPATH="$PWD/src"
export TORCH_CUDA_ARCH_LIST=8.9
export LUNA_GPU_LOG="$LUNA_WORK/logs/rtx4090-$SLURM_JOB_ID"
mkdir -p "$LUNA_WORK/outputs/gpu-$SLURM_JOB_ID"
echo "Allocation host: $(hostname); Slurm job: $SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\nLD_LIBRARY_PATH=%s\n' "${CUDA_VISIBLE_DEVICES:-}" "${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true
ls -l /dev/nvidia* || true
if ! "$LUNA_PYTHON" - <<'PY'
import json
import os
import socket
from pathlib import Path
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, 'This session must expose exactly one allocated GPU'
gpu = torch.cuda.get_device_properties(0)
assert '4090' in gpu.name, f'Expected RTX 4090, got {gpu.name}'
assert torch.cuda.get_device_capability(0) == (8, 9)
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              device=gpu.name, memory_bytes=gpu.total_memory,
              torch=torch.__version__, torch_cuda=torch.version.cuda,
              capability=list(torch.cuda.get_device_capability(0)))
path = Path(os.environ['LUNA_WORK']) / 'outputs' / f"gpu-{os.environ['SLURM_JOB_ID']}" / 'allocation.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY
then
    echo "CUDA check failed. This compute-node shell remains available for diagnosis."
fi
echo "CUDA compiler candidates on $(hostname):"
for luna_nvcc in /opt/ohpc/pub/apps/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc; do
    if [[ -x "$luna_nvcc" ]]; then
        echo "$luna_nvcc"
        "$luna_nvcc" --version
    fi
done
echo "Allocated GPU session ready; commands now run on $(hostname)."
export PS1="(LUNA $SLURM_JOB_ID \\h) \\w \\$ "
exec bash --noprofile --norc -i
