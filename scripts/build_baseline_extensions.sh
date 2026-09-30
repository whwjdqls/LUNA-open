#!/usr/bin/env bash
# Build the native LHM dependencies on a Slurm CPU allocation, targeting B200.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/parcc_env.sh
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run native extension compilation inside a Slurm allocation."
  exit 1
fi
source /usr/share/lmod/lmod/init/bash
module load cuda/12.8.1
export TORCH_CUDA_ARCH_LIST=10.0
export FORCE_CUDA=1
export MAX_JOBS=4
export TORCH_EXTENSIONS_DIR="$LUNA_WORK/cache/torch_extensions_baselines"
mkdir -p "$TORCH_EXTENSIONS_DIR"
baseline_python="$LUNA_WORK/envs/lhm/bin/python"
references="$LUNA_WORK/references"
"$baseline_python" -m pip check
# Verify source pins before any build modifies generated/ignored files.
"$LUNA_PYTHON" scripts/checkout_references.py --root "$references" \
  --repos pytorch3d-baseline basicsr-baseline diff-gaussian-baseline
"$baseline_python" -m pip install --no-build-isolation --no-deps \
  "$references/pytorch3d-baseline"
"$baseline_python" -m pip install --no-build-isolation --no-deps \
  "$references/diff-gaussian-baseline"
# BasicSR's selected RRDB network uses ordinary PyTorch layers. Its optional
# deform-convolution/fused-activation extensions are not needed by this network.
"$baseline_python" -m pip install --no-build-isolation \
  -c requirements-baselines.txt "$references/basicsr-baseline"
"$baseline_python" -m pip check
PYTHONPATH="$references/LHM" "$baseline_python" - <<'PY'
import json
import sys

import diff_gaussian_rasterization
import pytorch3d
import torch
import torchvision
import xformers
from basicsr.archs.rrdbnet_arch import RRDBNet
from pytorch3d.ops import knn_points

import LHM.models.modeling_human_lrm

points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
nearest = knn_points(points, points, K=1)
torch.testing.assert_close(nearest.dists, torch.zeros(1, 2, 1))
assert nearest.idx.tolist() == [[[0], [1]]]
print(json.dumps(dict(
    environment=sys.prefix,
    torch=torch.__version__, torchvision=torchvision.__version__,
    pytorch3d=pytorch3d.__version__, xformers=xformers.__version__,
    native_lhm_import=True, cpu_knn=True,
    limitation="Import and CPU extension check only; no native checkpoint or GPU inference yet",
), indent=2))
PY
