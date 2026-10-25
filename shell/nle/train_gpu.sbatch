#!/bin/bash
# One surrogate training on one GPU from a source snapshot (shell/nle/snapshot.sh).
#   sbatch -A alice_g -C gpu -q shared -t 03:00:00 --gpus 1 -c 32 \
#     shell/nle/train_gpu.sbatch SNAPSHOT_DIR RUN_DIR [train args...]
#SBATCH -N 1
#SBATCH -J nle-train
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; RUN=$2; shift 2
export PYTHONPATH=$SNAP/src
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
mkdir -p "$RUN"
echo "job $SLURM_JOB_ID node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" | tee -a "$RUN/job.log"
nvidia-smi -L | tee -a "$RUN/job.log"
$PY -c "import omnilearned, sys; assert omnilearned.__file__.startswith('$SNAP'), omnilearned.__file__"
$PY -m omnilearned.sbi.toy.train --out "$RUN" "$@"
echo "end $(date -Is)" | tee -a "$RUN/job.log"
