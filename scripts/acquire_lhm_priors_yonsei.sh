#!/usr/bin/env bash
# Acquire the public prior archive linked by the pinned LHM README.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Slurm allocation required' >&2; exit 2; }
cd /home/whwjdqls99/LUNA-open
source scripts/yonsei_env.sh
lhm_prior="$LUNA_WORK/assets/lhm-priors"
mkdir -p "$lhm_prior"
exec > >(tee -a "$lhm_prior/acquisition.log") 2>&1
printf 'Host=%s Job=%s Date=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$(date -Is)"
exec 9>"$lhm_prior/acquisition.lock"
flock --nonblock 9
lhm_url=https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar
if [[ ! -f "$lhm_prior/LHM_prior_model.tar" ]]; then
    curl --fail --location --retry 5 --continue-at - --output "$lhm_prior/LHM_prior_model.tar.partial" "$lhm_url"
    mv "$lhm_prior/LHM_prior_model.tar.partial" "$lhm_prior/LHM_prior_model.tar"
fi
sha256sum "$lhm_prior/LHM_prior_model.tar" > "$lhm_prior/archive.sha256"
tar -tf "$lhm_prior/LHM_prior_model.tar" > "$lhm_prior/archive-contents.txt"
if [[ ! -f "$lhm_prior/extraction-completed.txt" ]]; then
    tar --no-same-owner -xf "$lhm_prior/LHM_prior_model.tar" -C "$lhm_prior"
    date -Is > "$lhm_prior/extraction-completed.txt"
fi
printf '%s\n' "$lhm_url" > "$lhm_prior/source-url.txt"
date -Is > "$lhm_prior/acquisition-completed.txt"
