#!/usr/bin/env bash
# Isolated native LHM inference setup; execute only on a Slurm compute node.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export FORCE_CUDA=1
export MAX_JOBS=4
export PIP_DEFAULT_TIMEOUT=180
export PIP_RETRIES=8
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/lhm-torch23-cu118"
lhm_python="$LUNA_WORK/envs/lhm-native/bin/python"
lhm_run="$LUNA_WORK/baselines/lhm-20260928"
mkdir -p "$lhm_run" "$LUNA_WORK/assets/lhm-priors"
exec > >(tee -a "$lhm_run/setup.log") 2>&1
printf 'Host=%s Job=%s Date=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$(date -Is)"
exec 9>"$lhm_run/setup.lock"
flock --nonblock 9
if [[ ! -f "$LUNA_WORK/envs/lhm-native/pyvenv.cfg" ]]; then
    /home/whwjdqls99/miniconda3/envs/robocasa/bin/python -m venv "$LUNA_WORK/envs/lhm-native"
fi
# Reuse exact upstream Torch/torchvision versions already installed locally.
# Copy packages into the private environment; leave the source environment intact.
if [[ ! -f "$lhm_run/copied-torch-packages.json" ]]; then
    "$LUNA_DATA_PYTHON" - <<'PY'
import json
import shutil
from pathlib import Path
origin = Path('/home/whwjdqls99/miniconda3/envs/robocasa/lib/python3.10/site-packages')
target = Path('/scratch2/whwjdqls99/LUNA-open/envs/lhm-native/lib/python3.10/site-packages')
patterns = ['torch', 'torchgen', 'torch-*.dist-info', 'torchvision', 'torchvision.libs',
            'torchvision-*.dist-info', 'triton', 'triton-*.dist-info', 'nvidia',
            'nvidia*.dist-info', 'numpy', 'numpy.libs', 'numpy-*.dist-info']
copied = []
for pattern in patterns:
    for source in origin.glob(pattern):
        shutil.copytree(source, target / source.name, dirs_exist_ok=True)
        copied.append(str(source))
Path('/scratch2/whwjdqls99/LUNA-open/baselines/lhm-20260928/copied-torch-packages.json').write_text(
    json.dumps(dict(origin=str(origin), copied=copied), indent=2) + '\n')
PY
fi
"$LUNA_DATA_PYTHON" scripts/copy_lhm_local_dependencies.py
"$lhm_python" -m pip install --disable-pip-version-check 'pip<25' 'setuptools<75' wheel
"$lhm_python" -m pip install 'numpy==1.23.5' 'torch==2.3.0' 'torchvision==0.18.0' \
    'xformers==0.0.26.post1' --extra-index-url https://download.pytorch.org/whl/cu118
"$lhm_python" -m pip install 'numpy==1.23.5' 'accelerate==0.31.0' 'diffusers==0.32.0' \
    'transformers==4.41.2' 'huggingface_hub==0.27.1' 'kornia==0.7.2' \
    'omegaconf==2.3.0' 'gsplat==1.4.0' 'timm==1.0.15' \
    'smplx==0.1.28' 'trimesh==4.4.9' 'scipy==1.13.1' 'opencv-python==4.11.0.86' \
    'Pillow==10.4.0' einops roma plyfile 'jaxtyping==0.2.38' 'typeguard==2.13.3' \
    'scikit-image==0.24.0' 'numba==0.59.1' 'llvmlite==0.42.0' \
    loguru lpips ninja packaging imageio imageio-ffmpeg safetensors \
    'basicsr==1.4.2' 'gfpgan==1.3.8' pyrender 'kiui==0.2.14'
"$lhm_python" -m pip install --no-build-isolation chumpy
"$lhm_python" -m pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
    'git+https://github.com/ashawkey/diff-gaussian-rasterization.git@8829d14f814fccdaf840b7b0f3021a616583c0a1'
"$lhm_python" -m pip freeze > "$lhm_run/requirements-resolved.txt"
"$lhm_python" -m pip check
if [[ ! -d "$lhm_run/source" ]]; then
    cp -a "$LUNA_WORK/references/LHM" "$lhm_run/source"
fi
date -Is > "$lhm_run/setup-completed.txt"
