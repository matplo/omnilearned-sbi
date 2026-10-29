#!/bin/bash
# Selection-acceptance tables of one checkpoint on one 4-GPU node: Pr_q(S | theta) and the
# per-N table Pr_q(S | N, theta) (fidelity plan S1) from the same common-random-number
# samples, 4 strided shards, then the merge. Axes as the M5 closures (phase 0).
#   sbatch -A alice_g -C gpu -q regular -t 01:30:00 -N 1 --gpus 4 \
#     shell/nle/fid_acceptance.sbatch SNAPSHOT_DIR CHECKPOINT OUT_DIR
#SBATCH -J nle-fidacc
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; CKPT=$2; OUT=$3
AXES=${ACC_AXES:-0:11:6 0.25:0.55:4 0.2:1.2:4}; NS=${ACC_SAMPLES:-8192}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1
echo "job $SLURM_JOB_ID node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") ckpt $CKPT start $(date -Is)" >> "$OUT/job.log"
AX=""; for x in $AXES; do AX="$AX --axis $x"; done
for k in 0 1 2 3; do
  [ -f "$OUT/part_$k.npz" ] && continue
  CUDA_VISIBLE_DEVICES=$k $PY -m omnilearned.sbi.toy.closure acceptance --checkpoint "$CKPT" $AX \
    --n-samples "$NS" --shard $k --num-shards 4 --device cuda --out "$OUT/part_$k.npz" > "$OUT/part_$k.log" 2>&1 &
done
wait
$PY -m omnilearned.sbi.toy.closure acceptance-merge --parts "$OUT"/part_{0,1,2,3}.npz --out "$OUT/acceptance.npz" \
  --per-n-out "$OUT/acceptance_per_n.npz" | tee -a "$OUT/job.log"
echo "end $(date -Is)" >> "$OUT/job.log"
