#!/bin/bash
# M3 gate of one checkpoint on one GPU node: 4 processes (one per GPU) over strided point
# shards, then the summary. Resumable (finished points are skipped).
#   sbatch -A alice_g -C gpu -q regular -t 02:30:00 -N 1 --gpus 4 \
#     shell/nle/gate_gpu.sbatch SNAPSHOT_DIR CHECKPOINT FIDELITY_DIR OUT_DIR [gate run args...]
#SBATCH -J nle-gate
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; CKPT=$2; FID=$3; OUT=$4; shift 4
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1
echo "job $SLURM_JOB_ID node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$OUT/job.log"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k $PY -m omnilearned.sbi.toy.gate run --checkpoint "$CKPT" \
    --fidelity-dir "$FID" --out "$OUT" --shard $k --num-shards 4 --device cuda "$@" \
    > "$OUT/gate_shard$k.log" 2>&1 &
done
wait
$PY -m omnilearned.sbi.toy.gate summarize --out "$OUT" --n-settings "${GATE_POINTS:-128}" | tee "$OUT/summary.txt"
echo "end $(date -Is)" >> "$OUT/job.log"
