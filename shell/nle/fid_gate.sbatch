#!/bin/bash
# Full M3 measurement of one checkpoint on one 4-GPU node, as in round 2: equivalent shift
# (GPU 0) and sampler convergence (GPU 1) first, then the gate at all validation points over
# 4 strided shards, then the summary. Resumable (finished points and outputs are skipped).
#   sbatch -A alice_g -C gpu -q regular -t 02:30:00 -N 1 --gpus 4 \
#     shell/nle/fid_gate.sbatch SNAPSHOT_DIR CHECKPOINT OUT_DIR [gate run args...]
#SBATCH -J nle-fidgate
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; CKPT=$2; OUT=$3; shift 3
FID=${FID:-$JEST_NLE_ROOT/data/v2_m1_fidelity}; SCAN=${SCAN:-$JEST_NLE_ROOT/data/v2_m1_devscan}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT/gate"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
$PY -c "import omnilearned; assert omnilearned.__file__.startswith('$SNAP'), omnilearned.__file__"
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1
echo "job $SLURM_JOB_ID node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") ckpt $CKPT sha256 $(sha256sum "$CKPT" | cut -c1-16) start $(date -Is)" >> "$OUT/job.log"
nvidia-smi -L >> "$OUT/job.log"
for k in 0 1 2 3; do
  ( export CUDA_VISIBLE_DEVICES=$k
    if [ $k = 0 ] && [ ! -f "$OUT/shift.json" ]; then
      $PY -m omnilearned.sbi.toy.gate shift --checkpoint "$CKPT" --scan-dir "$SCAN" \
        --out "$OUT/shift.json" --device cuda >> "$OUT/shift.log" 2>&1
    fi
    if [ $k = 1 ] && [ ! -f "$OUT/convergence.json" ]; then
      $PY -m omnilearned.sbi.toy.gate convergence --checkpoint "$CKPT" --fidelity-dir "$FID" \
        --out "$OUT/convergence.json" --points 0 64 --steps 64 128 256 --n-generate 8192 \
        --device cuda >> "$OUT/convergence.log" 2>&1
    fi
    $PY -m omnilearned.sbi.toy.gate run --checkpoint "$CKPT" --fidelity-dir "$FID" --out "$OUT/gate" \
      --shard $k --num-shards 4 --device cuda "$@" >> "$OUT/gate_shard$k.log" 2>&1 ) &
done
wait
$PY -m omnilearned.sbi.toy.gate summarize --out "$OUT/gate" --n-settings "${GATE_POINTS:-128}" > "$OUT/summary.txt"
echo "end $(date -Is)" >> "$OUT/job.log"
