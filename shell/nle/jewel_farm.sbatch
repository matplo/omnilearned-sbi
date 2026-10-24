#!/bin/bash
# JEWEL generation farm on CPU nodes: WORKERS single-threaded workers per node claim tasks of
# a task list (omnilearned.sbi.jewel.generate plan) until none is left. Resumable; job arrays
# add nodes. Production needs a user-approved budget (plan rule 8).
#   sbatch -A alice -C cpu -q regular -t 04:00:00 -N 1 --array=0-NODES-1 \
#     shell/nle/jewel_farm.sbatch SNAPSHOT TASK_DIR [WORKERS] [TIMEOUT_S]
#SBATCH -J nle-jewel
#SBATCH -o %x-%A_%a.out
set -uo pipefail
SNAP=$1; DIR=$2; W=${3:-128}; TO=${4:-14400}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
SCR=${TMPDIR:-/tmp}/jewel-$USER-${SLURM_JOB_ID:-local}
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$SCR" "$DIR/logs"
RUN="henv -n omni-sbi -x python"
echo "job ${SLURM_JOB_ID:-local} node $(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) workers $W snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$DIR/logs/jobs.log"
$RUN -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" > /dev/null 2>&1  # cppyy PCH, once per node
[ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ] && $RUN -m omnilearned.sbi.jewel.generate unlock-stale --dir "$DIR"
for i in $(seq 1 "$W"); do
  $RUN -m omnilearned.sbi.jewel.generate work --dir "$DIR" --scratch "$SCR" --timeout "$TO" \
    > "$DIR/logs/worker_${SLURM_JOB_ID:-local}_$i.log" 2>&1 &
done
wait
# The index hashes every shard (28 k files: ~30 min on CFS, CPU idle); by default run it once
# afterwards from a login node (generate index --dir DIR). INDEX=1 makes array element 0 do it.
if [ "${INDEX:-0}" = 1 ] && [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  $RUN -m omnilearned.sbi.jewel.generate index --dir "$DIR" | tee -a "$DIR/logs/jobs.log"
fi
echo "end $(hostname) $(date -Is)" >> "$DIR/logs/jobs.log"
rm -rf "$SCR"
