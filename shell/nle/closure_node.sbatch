#!/bin/bash
# Closure campaign fits on GPU nodes: 4 GPUs per node x PER_GPU processes per GPU, one fit
# per process at a time; a job array spreads the campaign over nodes. Resumable (sealed fits
# are skipped), so a resubmission only runs what is missing.
#   sbatch -A alice_g -C gpu -q regular -t 04:00:00 -N 1 --gpus 4 --array=0-NODES-1 \
#     shell/nle/closure_node.sbatch SNAPSHOT CAMPAIGN_DIR [PER_GPU]
#SBATCH -J nle-closure
#SBATCH -o %x-%A_%a.out
set -euo pipefail
SNAP=$1; DIR=$2; PER=${3:-1}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
PY=${PY:-$VIRTUAL_ENV/bin/python}
NODE=${SLURM_ARRAY_TASK_ID:-0}; NODES=${SLURM_ARRAY_TASK_COUNT:-1}
W=$((NODES * 4 * PER))
mkdir -p "$DIR/logs"
echo "job ${SLURM_JOB_ID:-local} node $(hostname) index $NODE/$NODES per_gpu $PER snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$DIR/logs/jobs.log"
for g in 0 1 2 3; do
  for p in $(seq 0 $((PER - 1))); do
    w=$(( (NODE * 4 + g) * PER + p ))
    CUDA_VISIBLE_DEVICES=$g "$PY" -m omnilearned.sbi.toy.closure fit --dir "$DIR" \
      --worker $w --num-workers $W > "$DIR/logs/worker_$w.log" 2>&1 &
  done
done
wait
echo "end $(hostname) $(date -Is)" >> "$DIR/logs/jobs.log"
