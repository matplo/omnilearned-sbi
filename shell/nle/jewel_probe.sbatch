#!/bin/bash
# JEWEL probe: runs every task of the given task directories at once on one CPU node.
#   sbatch -A alice -C cpu -q debug -t 00:30:00 -N 1 shell/nle/jewel_probe.sbatch SNAPSHOT TIMEOUT_S DIR [DIR...]
#SBATCH -J nle-jewel-probe
#SBATCH -o %x-%j.out
set -uo pipefail
SNAP=$1; TO=$2; shift 2
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
SCR=${TMPDIR:-/tmp}/jewel-$USER-${SLURM_JOB_ID:-local}
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$SCR"
RUN="henv -n omni-sbi -x python"
$RUN -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" > /dev/null 2>&1
for d in "$@"; do
  mkdir -p "$d/logs"
  echo "job ${SLURM_JOB_ID:-local} node $(hostname) snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$d/logs/jobs.log"
  n=$($RUN -c "import json,sys; print(len(json.load(open('$d/tasks.json'))['tasks']))")
  for i in $(seq 1 "$n"); do
    $RUN -m omnilearned.sbi.jewel.generate work --dir "$d" --scratch "$SCR" --timeout "$TO" --max-tasks 1 \
      > "$d/logs/worker_${SLURM_JOB_ID:-local}_$i.log" 2>&1 &
  done
done
wait
for d in "$@"; do $RUN -m omnilearned.sbi.jewel.generate index --dir "$d" | tee -a "$d/logs/jobs.log"; done
rm -rf "$SCR"
