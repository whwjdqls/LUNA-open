#!/usr/bin/env bash
# Compare the prepared teacher against the pinned NeuMan and SMPL-X definitions.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "A Slurm allocation is required." >&2; exit 2; }
cd "${LUNA_REPO:-${SLURM_SUBMIT_DIR:-/home/whwjdqls99/LUNA-open}}"
source scripts/yonsei_env.sh
echo "SMPL audit on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
reference="$LUNA_WORK/references/ml-neuman"
revision=15d64ac218b1c8bd6a99ab876d2408898c859c69
if [[ ! -e "$reference" ]]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/apple-aiml-research/ml-neuman.git "$reference"
    git -C "$reference" checkout --detach "$revision"
fi
[[ "$(git -C "$reference" rev-parse HEAD)" == "$revision" ]] || {
    echo "Existing NeuMan source differs from the required pin." >&2; exit 2;
}
"$LUNA_PYTHON" scripts/audit_smpl.py \
    --root "$LUNA_WORK/data/neuman/dataset" \
    --manifest "$LUNA_WORK/data/neuman/manifest-v2.json" \
    --smpl "$LUNA_WORK/assets/smpl/SMPL_NEUTRAL.pkl" \
    --neuman-code "$reference" --output "$LUNA_WORK/outputs/smpl-audit-$SLURM_JOB_ID"
echo "SMPL audit completed at $(date -Is)"
