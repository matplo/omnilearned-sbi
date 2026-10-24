#!/bin/bash
# One chunk of M3 gating for two checkpoints inside a 4-GPU allocation: each model gets two
# GPUs (shard 0 also runs sampler convergence and the equivalent shift first, shard 1 gates
# right away). Resumable: finished gate points, convergence.json and shift.json are skipped.
#   shell/nle/gate_chunk.sh SNAPSHOT CKPT_A OUT_A CKPT_B OUT_B
set -uo pipefail
SNAP=$1; CA=$2; OA=$3; CB=$4; OB=$5
HERE=$(cd "$(dirname "$0")" && pwd)
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export VIRTUAL_ENV=${VIRTUAL_ENV:-/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/.venv}
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER/allDict.cxx.pch PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OA" "$OB"
PYTHONPATH=$SNAP/src "$VIRTUAL_ENV/bin/python" -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" > /dev/null 2>&1
echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] gate chunk start, snapshot $(cat "$SNAP/SOURCE_REVISION")" | tee -a "$OA/post_train.log" "$OB/post_train.log"
bash "$HERE/post_train.sh" "$SNAP" "$CA" "$OA" 0 0 2 &
bash "$HERE/post_train.sh" "$SNAP" "$CA" "$OA" 1 1 2 skip &
bash "$HERE/post_train.sh" "$SNAP" "$CB" "$OB" 2 0 2 &
bash "$HERE/post_train.sh" "$SNAP" "$CB" "$OB" 3 1 2 skip &
wait
echo "[$(date -Is) $(hostname)] gate chunk end: $(ls "$OA"/gate/point_*.json 2>/dev/null | wc -l) and $(ls "$OB"/gate/point_*.json 2>/dev/null | wc -l) of 128 points" | tee -a "$OA/post_train.log" "$OB/post_train.log"
