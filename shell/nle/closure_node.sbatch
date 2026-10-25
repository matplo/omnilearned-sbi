#!/bin/bash
# Closure campaign fits on GPU nodes: 4 GPUs per node x PER_GPU processes per GPU, one fit
# per process at a time; workers claim unsealed fits (lock files), so job arrays, several
# jobs and interactive sessions can share one campaign. Fits resume from their cache.
#   sbatch -A alice_g -C gpu -q regular -t 04:00:00 -N 1 --gpus 4 --array=0-NODES-1 \
#     shell/nle/closure_node.sbatch SNAPSHOT CAMPAIGN_DIR [PER_GPU]
#SBATCH -J nle-closure
#SBATCH -o %x-%A_%a.out
set -euo pipefail
SNAP=$1; DIR=$2; PER=${3:-1}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
NODE=${SLURM_ARRAY_TASK_ID:-0}; NODES=${SLURM_ARRAY_TASK_COUNT:-1}
W=$((NODES * 4 * PER))
mkdir -p "$DIR/logs"
# stale claims (ended jobs) are removed by this one process, before any worker starts
[ "$NODE" -eq 0 ] && $PY -m omnilearned.sbi.toy.closure unlock-stale --dir "$DIR"
echo "job ${SLURM_JOB_ID:-local} node $(hostname) index $NODE/$NODES per_gpu $PER snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$DIR/logs/jobs.log"
for g in 0 1 2 3; do
  for p in $(seq 0 $((PER - 1))); do
    w=$(( (NODE * 4 + g) * PER + p ))
    CUDA_VISIBLE_DEVICES=$g $PY -m omnilearned.sbi.toy.closure fit --dir "$DIR" --claim \
      > "$DIR/logs/worker_${SLURM_JOB_ID:-local}_$w.log" 2>&1 &
  done
done
wait
echo "end $(hostname) $(date -Is)" >> "$DIR/logs/jobs.log"
