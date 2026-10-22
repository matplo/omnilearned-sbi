#!/bin/bash
# After a training: sampler convergence (lesson 5b), equivalent shift at the development
# setting and along the tube, then gate points (strided shard) on one GPU. Resumable gate.
#   shell/nle/post_train.sh SNAPSHOT CHECKPOINT OUT GPU SHARD NUM_SHARDS [skip-conv]
set -euo pipefail
SNAP=$1; CKPT=$2; OUT=$3; GPU=$4; SHARD=$5; NSH=$6; SKIP=${7:-}
export PYTHONPATH=$SNAP/src CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER/allDict.cxx.pch
PY=${PY:-$VIRTUAL_ENV/bin/python}
FID=$JEST_NLE_ROOT/data/v2_m1_fidelity; SCAN=$JEST_NLE_ROOT/data/v2_m1_devscan
mkdir -p "$OUT"
echo "post_train $(date -Is) snapshot $(cat "$SNAP/SOURCE_REVISION") ckpt $CKPT gpu $GPU shard $SHARD/$NSH" >> "$OUT/post_train.log"
if [ -z "$SKIP" ]; then
  [ -f "$OUT/convergence.json" ] || "$PY" -m omnilearned.sbi.toy.gate convergence --checkpoint "$CKPT" \
    --fidelity-dir "$FID" --out "$OUT/convergence.json" --points 0 64 --steps 64 128 256 \
    --n-generate 8192 --device cuda >> "$OUT/convergence.log" 2>&1
  [ -f "$OUT/shift.json" ] || "$PY" -m omnilearned.sbi.toy.gate shift --checkpoint "$CKPT" \
    --scan-dir "$SCAN" --out "$OUT/shift.json" --device cuda >> "$OUT/shift.log" 2>&1
fi
"$PY" -m omnilearned.sbi.toy.gate run --checkpoint "$CKPT" --fidelity-dir "$FID" --out "$OUT/gate" \
  --shard "$SHARD" --num-shards "$NSH" --device cuda >> "$OUT/gate_shard$SHARD.log" 2>&1
echo "post_train done $(date -Is) gpu $GPU shard $SHARD" >> "$OUT/post_train.log"
