#!/usr/bin/env bash
# Separate LHM++ package overlay; preserve the completed LHM and training envs.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm compute node required' >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export FORCE_CUDA=1
export MAX_JOBS=4
export PIP_DEFAULT_TIMEOUT=120
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
lhmpp_base="$LUNA_WORK/baselines/lhmpp-20260928"
lhmpp_env="$LUNA_WORK/envs/lhmpp-native"
mkdir -p "$lhmpp_base"
exec > >(tee -a "$lhmpp_base/setup.log") 2>&1
printf 'Host=%s Job=%s Date=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$(date -Is)"
if [[ ! -f "$lhmpp_env/pyvenv.cfg" ]]; then
    "$LUNA_WORK/envs/lhm-native/bin/python" -m venv "$lhmpp_env"
    printf '%s\n' "$LUNA_WORK/envs/lhm-native/lib/python3.10/site-packages" > "$lhmpp_env/lib/python3.10/site-packages/lhm_base.pth"
fi
"$lhmpp_env/bin/python" - <<'PY'
import json, shutil
from pathlib import Path
w = Path('/scratch2/whwjdqls99/LUNA-open')
r = w / 'baselines/lhmpp-20260928/source'
if not r.exists():
    shutil.copytree(w / 'references/LHM-plusplus', r)
links = {r / 'pretrained_models': w / 'assets/lhm-priors/pretrained_models'}
for target, source in links.items():
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)
    elif target.resolve() != source.resolve() and not (
        target.is_dir() and (target / 'human_model_files').resolve() == (source / 'human_model_files').resolve()
    ):
        raise ValueError(target)
(r.parent / 'asset-links.json').write_text(json.dumps({str(k): str(v) for k,v in links.items()}, indent=2))
PY
"$lhmpp_env/bin/python" -m pip install 'spconv-cu118==2.3.8' addict
"$lhmpp_env/bin/python" -m pip install --no-index 'torch-scatter==2.1.2+pt23cu118' -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
"$lhmpp_env/bin/python" -m pip install --no-build-isolation 'flash-attn==2.6.3'
"$lhmpp_env/bin/python" -m pip install --no-build-isolation "$lhmpp_base/source/lib/pointops"
"$lhmpp_env/bin/python" -m pip freeze > "$lhmpp_base/requirements-resolved.txt"
"$lhmpp_env/bin/python" -m pip check
date -Is > "$lhmpp_base/setup-completed.txt"
