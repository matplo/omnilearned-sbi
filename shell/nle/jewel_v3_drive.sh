#!/bin/bash
# Drive one JEWEL v3 split to a jet target, unattended, from a login node.
#
#   nohup shell/nle/jewel_v3_drive.sh SNAP SPLIT TARGET [NODES] [WAVES] > LOG 2>&1 &
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
WEXPO=${WEXPO:-3}            # pT-hat sampling exponent; 3 = log-flat over 10-600 GeV
DATASET=${DATASET:-jewel_v3} # data/<DATASET>/<split>
# NEVENT: events per task. randomness.chunks_per_point caps a point at 100 tasks (NJOB is
# base + 100*point + chunk and cards.njob refuses chunk >= 100), so TARGET jets per point
# must be reachable in <= 100 tasks. At the measured 0.40 jets/event, 1000 events/task gives
# ~400 jets/task = 40k jets/point -- enough for train/fidelity (20k) and validation (5k), but
# NOT for the test split's 50k. Pass NEVENT=5000 there.
NEVENT=${NEVENT:-}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
D=$JEST_NLE_ROOT/data/$DATASET/$SPLIT
CFG=$SNAP/configs/jewel-2.6.0-cs-v3.toml
RUN="henv -n omni-sbi -x python"
GEN="$RUN -m omnilearned.sbi.jewel.generate"
export PYTHONPATH=$SNAP/src
mkdir -p "$D/logs"
log() { echo "[$(date -Is) drive $SPLIT] $*" | tee -a "$D/logs/drive.log"; }

log "start: snapshot $(cat "$SNAP/SOURCE_REVISION") dataset ${DATASET} wexpo ${WEXPO} target ${TARGET} jets/point, ${NODES} nodes x ${WORKERS} workers, <= ${WAVES} waves"

if [ ! -f "$D/tasks.json" ]; then
  log "plan $SPLIT"
  # NO --no-events: v3 always writes raw event stores, so a future representation change
  # never needs a regeneration. v2 passed it and locked 10.3 node-h to one representation.
  $GEN plan --split "$SPLIT" --out "$D" --config "$CFG" --wexpo "$WEXPO" --chunks 2 \
    ${NEVENT:+--nevent "$NEVENT"} 2>&1 | tee -a "$D/logs/drive.log"
  [ -f "$D/tasks.json" ] || { log "plan FAILED"; exit 1; }
fi

done_tasks() { ls "$D"/${SPLIT}_p*.json 2>/dev/null | wc -l; }
all_tasks() { $RUN -c "import json;print(len(json.load(open('$D/tasks.json'))['tasks']))"; }

for w in $(seq 1 "$WAVES"); do
  before=$(done_tasks); total=$(all_tasks)
  if [ "$before" -ge "$total" ] && [ "$w" -gt 1 ]; then log "all $total tasks done"; break; fi
  log "wave $w/$WAVES: $before/$total tasks done, submitting ${NODES}-node farm"
  JID=$(sbatch --parsable -A alice -C cpu -q regular -t 04:00:00 -N 1 \
        --array=0-$((NODES - 1)) -J "jv3-$SPLIT" \
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

log "index + merge (compute node: merge_v3 loads a whole group into memory, which OOM-killed"
log "  the validation merge -- 1392 shards, 649789 jets -- on the login node, 2026-11-01)"
MJID=$(sbatch --parsable -A alice -C cpu -q debug -t 00:30:00 -N 1 \
      -o "$D/logs/merge-%j.log" -J "v3merge-$SPLIT" \
      "$(dirname "$0")/jewel_v3_merge.sbatch" "$SNAP" "$D" "$JEST_NLE_ROOT/data/$DATASET/merged" "$SPLIT" "$CFG")
if [ -n "$MJID" ]; then
  log "merge job $MJID submitted, waiting"
  while squeue -h -j "$MJID" 2>/dev/null | grep -q .; do sleep 60; done
  tail -5 "$D/logs/merge-$MJID.log" | tee -a "$D/logs/drive.log"
else
  log "merge sbatch FAILED to submit"
fi
if [ -f "$JEST_NLE_ROOT/data/$DATASET/merged/index_${SPLIT}.json" ]; then
  log "DONE: merged/index_${SPLIT}.json written"
  touch "$D/DONE"
else
  log "merge did NOT produce index_${SPLIT}.json -- inspect $D/logs/"
fi
