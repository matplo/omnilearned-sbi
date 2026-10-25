#!/bin/bash
# Boundary / N-mix diagnostic (toy.boundary) of one checkpoint on one 4-GPU node: the gate
# points POINTS (default every 4th of 128) over 4 strided shards, then the summary.
#   sbatch -A alice_g -C gpu -q regular -t 01:30:00 -N 1 --gpus 4 \
#     shell/nle/fid_boundary.sbatch SNAPSHOT_DIR CHECKPOINT OUT_DIR
#SBATCH -J nle-fidbnd
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; CKPT=$2; OUT=$3
FID=${FID:-$JEST_NLE_ROOT/data/v2_m1_fidelity}
POINTS=(${POINTS:-$(seq 0 4 127)})
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1
echo "job $SLURM_JOB_ID node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") ckpt $CKPT start $(date -Is)" >> "$OUT/job.log"
for k in 0 1 2 3; do
  P=$(for ((i = k; i < ${#POINTS[@]}; i += 4)); do echo -n "${POINTS[$i]} "; done)
  CUDA_VISIBLE_DEVICES=$k $PY -m omnilearned.sbi.toy.boundary run --checkpoint "$CKPT" --fidelity-dir "$FID" \
    --points $P --out "$OUT" --device cuda > "$OUT/shard_$k.log" 2>&1 &
done
wait
$PY -m omnilearned.sbi.toy.boundary summarize --dir "$OUT" | tee "$OUT/summary.txt"
echo "end $(date -Is)" >> "$OUT/job.log"
