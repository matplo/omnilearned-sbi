#!/bin/bash
# Drive one JEWEL v2 split to a jet target, unattended, from a login node.
#
#   nohup shell/nle/jewel_v2_drive.sh SNAP SPLIT TARGET [NODES] [WAVES] > LOG 2>&1 &
#
# Each wave: submit a CPU farm array -> wait -> `unlock-stale` -> `topup` from the *measured*
# per-point acceptance -> repeat until every point has TARGET jets, WAVES is reached, or a
# wave adds nothing. Self-correcting, so no acceptance has to be predicted up front, and
# resumable: workers skip tasks that already have a manifest, so re-running is free.
#
# Why a driver and not `--dependency=afterok`: a farm array element that hits its time limit
# is killed mid-task and leaves claim locks behind. `afterok` then never fires and the whole
# chain stalls silently (this cost the weighted run a full chain, 2026-10-27). The driver
# runs `unlock-stale` after *every* wave, so a timed-out element costs one wave, not the run.
#
# WORKERS defaults to 64, not the farm's 128. Inclusive clustering packs ~77 k jet candidates
# per task even after the sub-threshold pre-filter, ~1.9 GB per worker at max_constituents
# 512; 128 workers would sit near a 512 GB node's limit. Production is only a few node-hours
# of work, so it is not CPU-bound and the headroom is free.
set -uo pipefail
SNAP=$1; SPLIT=$2; TARGET=$3; NODES=${4:-8}; WAVES=${5:-4}; WORKERS=${6:-64}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
D=$JEST_NLE_ROOT/data/jewel_v2/$SPLIT
CFG=$SNAP/configs/jewel-2.6.0-cs-v2.toml
RUN="henv -n omni-sbi -x python"
GEN="$RUN -m omnilearned.sbi.jewel.generate"
export PYTHONPATH=$SNAP/src
mkdir -p "$D/logs"
log() { echo "[$(date -Is) drive $SPLIT] $*" | tee -a "$D/logs/drive.log"; }

log "start: snapshot $(cat "$SNAP/SOURCE_REVISION") target ${TARGET} jets/point, ${NODES} nodes x ${WORKERS} workers, <= ${WAVES} waves"

if [ ! -f "$D/tasks.json" ]; then
  log "plan $SPLIT"
  $GEN plan --split "$SPLIT" --out "$D" --config "$CFG" --wexpo 5 --chunks 2 --no-events \
    2>&1 | tee -a "$D/logs/drive.log"
  [ -f "$D/tasks.json" ] || { log "plan FAILED"; exit 1; }
fi

done_tasks() { ls "$D"/${SPLIT}_p*.json 2>/dev/null | wc -l; }
all_tasks() { $RUN -c "import json;print(len(json.load(open('$D/tasks.json'))['tasks']))"; }

for w in $(seq 1 "$WAVES"); do
  before=$(done_tasks); total=$(all_tasks)
  if [ "$before" -ge "$total" ] && [ "$w" -gt 1 ]; then log "all $total tasks done"; break; fi
  log "wave $w/$WAVES: $before/$total tasks done, submitting ${NODES}-node farm"
  JID=$(sbatch --parsable -A alice -C cpu -q regular -t 04:00:00 -N 1 \
        --array=0-$((NODES - 1)) -J "jv2-$SPLIT" \
        "$(dirname "$0")/jewel_farm.sbatch" "$SNAP" "$D" "$WORKERS" 14000)
  [ -z "$JID" ] && { log "sbatch FAILED"; break; }
  log "wave $w: job $JID submitted, waiting"
  while squeue -h -j "$JID" 2>/dev/null | grep -q .; do sleep 120; done
  after=$(done_tasks)
  log "wave $w: job $JID ended, tasks $before -> $after"

  # always, not only on failure: a killed element leaves locks that block everything after it
  $GEN unlock-stale --dir "$D" 2>&1 | tail -1 | tee -a "$D/logs/drive.log"

  if [ "$after" -le "$before" ] && [ "$w" -gt 1 ]; then
    log "wave $w added no tasks: stopping (see $D/logs/)"; break
  fi
  # grow the task list from the acceptance actually measured, then let the next wave run it
  $GEN topup --dir "$D" --target-jets "$TARGET" 2>&1 | tail -2 | tee -a "$D/logs/drive.log"
done

log "index + merge"
$GEN index --dir "$D" 2>&1 | tail -1 | tee -a "$D/logs/drive.log"
$GEN merge --dir "$D" --out "$JEST_NLE_ROOT/data/jewel_v2/merged" --shards 16 --workers 4 \
  2>&1 | tail -2 | tee -a "$D/logs/drive.log"
if [ -f "$JEST_NLE_ROOT/data/jewel_v2/merged/index_${SPLIT}.json" ]; then
  log "DONE: merged/index_${SPLIT}.json written"
  touch "$D/DONE"
else
  log "merge did NOT produce index_${SPLIT}.json -- inspect $D/logs/"
fi
