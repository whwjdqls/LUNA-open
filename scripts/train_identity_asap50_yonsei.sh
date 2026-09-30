#!/usr/bin/env bash
# Reuse the user's idle RTX4090 allocation, with an isolated, seed-matched run.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "${SLURMD_NODENAME:-}" != *login* ]] || exit 2
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=2
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export TORCH_CUDA_ARCH_LIST=8.9
export PATH="$CUDA_HOME/bin:$PATH"
asap_run="$LUNA_WORK/runs/neuman-identity-asap50-acap10-20260929"
asap_base="$LUNA_WORK/runs/neuman-identity-v2-20260928"
[[ ! -e "$asap_run" ]] || { echo "Run directory exists; inspect before resuming."; exit 2; }
mkdir -p "$asap_run/source" "$asap_run/provenance"
exec > >(tee -a "$asap_run/launch.log") 2>&1
echo "ASAP50/ACAP10 starts on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,utilization.gpu --format=csv
# The old run's executed source makes this a coefficient-only training ablation.
cp -a "$asap_base/source/src" "$asap_base/source/scripts" \
      "$asap_base/source/configs" "$asap_run/source/"
cp configs/neuman_yonsei_identity_asap50.yaml "$asap_run/source/configs/"
cp scripts/train_identity_asap50_yonsei.sh "$asap_run/source/scripts/"
cp docs/identity-asap50.md "$asap_run/provenance/experiment.md"
cp "$asap_base/identity/startup-000000.json" "$asap_run/provenance/baseline-startup.json"
tail -n 4 "$asap_base/launch.log" > "$asap_run/provenance/baseline-last-log.txt"
git rev-parse HEAD > "$asap_run/provenance/repository-head.txt"
git status --short > "$asap_run/provenance/repository-status.txt"
git diff --binary > "$asap_run/provenance/repository-working-tree.diff"
cd "$asap_run/source"
export PYTHONPATH="$PWD/src"
"$LUNA_PYTHON" - <<'PY'
import hashlib,json,os,socket
from pathlib import Path
import torch,yaml
root=Path.cwd().parent
baseline=json.loads((root/'provenance/baseline-startup.json').read_text())
cfg=yaml.safe_load(Path('configs/neuman_yonsei_identity_asap50.yaml').read_text())
comparison=json.loads(json.dumps(cfg))
comparison['output']=baseline['config']['output']
comparison['identity_loss']['anisotropy_weight']=baseline['config']['identity_loss']['anisotropy_weight']
assert comparison==baseline['config'], 'Unexpected additional configuration change'
assert cfg['identity_loss']['anisotropy_weight']==50 and cfg['identity_loss']['anchor_weight']==10
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert '4090' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0)==(8,9)
for name,expected in baseline['source_sha256'].items():
    assert hashlib.sha256((Path('src/luna_open')/name).read_bytes()).hexdigest()==expected,name
receipt=dict(job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),
    initialization='fresh, same seed and executed training source as identity v2',
    baseline_source_hashes_match=True,only_training_change='anisotropy_weight: 0.01 -> 50',
    asap_definition='mean relu(max_scale/min_scale - 5), denominator clamped at 1e-8',
    acap_definition='mean relu(distance from shaped SMPL anchor - 0.0525 meters)',
    asap_weight=50,acap_weight=10,requested_updates=cfg['training']['updates'])
(root/'provenance/launch-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
"$LUNA_PYTHON" -m luna_open.identity_training --config configs/neuman_yonsei_identity_asap50.yaml
echo "ASAP50/ACAP10 training finished at $(date -Is)"
