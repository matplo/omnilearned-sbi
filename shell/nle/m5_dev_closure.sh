#!/bin/bash
# One chunk of a closure campaign inside a GPU allocation (4 GPUs). Every phase is
# idempotent, so the same command continues where the previous chunk stopped:
#   0. selection-acceptance table Pr_q(S|theta) of the checkpoint (4 GPU shards, merged)
#   1. prepare the toy campaign and the surrogate self-closure control (once)
#   2. interim reports of whatever is sealed
#   3. fits: 4 GPUs x PER workers claim unsealed fits of both campaigns (interleaved);
#      fits resume from their cache after the allocation ends
#   4. final reports when everything is sealed; writes BASE/DONE
#   shell/nle/m5_dev_closure.sh BASE        (BASE holds m5.env, spec_toy.toml, spec_control.toml)
set -uo pipefail
BASE=$(cd "$1" && pwd)
# shellcheck disable=SC1091
source "$BASE/m5.env"   # SNAP, CKPT, ACC_AXES, ACC_SAMPLES, PER
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$BASE/logs"
RUN="henv -n omni-sbi -x python"
CL="$RUN -m omnilearned.sbi.toy.closure"
log() { echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] $*" | tee -a "$BASE/logs/progress.log"; }
nseal() { ls "$BASE"/toy/fits/*/seal.json "$BASE"/control/fits/*/seal.json 2>/dev/null | wc -l; }

log "chunk start, snapshot $(cat "$SNAP/SOURCE_REVISION"), sealed $(nseal)"
nvidia-smi -L >> "$BASE/logs/progress.log" 2>&1
log "cpus: nproc $(nproc), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}"
$RUN -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" > /dev/null 2>&1  # build the cppyy PCH once

# 0. acceptance table
if [ ! -f "$BASE/acceptance.npz" ]; then
  log "phase 0: acceptance table ($ACC_AXES, $ACC_SAMPLES clouds per point)"
  AX=""; for a in $ACC_AXES; do AX="$AX --axis $a"; done
  for g in 0 1 2 3; do
    [ -f "$BASE/acc_part_$g.npz" ] && continue
    CUDA_VISIBLE_DEVICES=$g $CL acceptance --checkpoint "$CKPT" $AX --n-samples "$ACC_SAMPLES" \
      --shard $g --num-shards 4 --device cuda --out "$BASE/acc_part_$g.npz" > "$BASE/logs/acc_$g.log" 2>&1 &
  done
  wait
  $CL acceptance-merge --parts "$BASE"/acc_part_{0,1,2,3}.npz --out "$BASE/acceptance.npz" 2>&1 | tail -1 | tee -a "$BASE/logs/progress.log"
  [ -f "$BASE/acceptance.npz" ] || { log "acceptance table failed"; exit 1; }
fi

# 1. prepare
for c in toy control; do
  if [ ! -f "$BASE/$c/campaign.json" ]; then
    log "phase 1: prepare $c"
    rm -rf "$BASE/$c"
    CUDA_VISIBLE_DEVICES=0 $CL prepare --spec "$BASE/spec_$c.toml" --out "$BASE/$c" 2>&1 | tail -1 | tee -a "$BASE/logs/progress.log"
    [ -f "$BASE/$c/campaign.json" ] || { log "prepare $c failed"; exit 1; }
  fi
done

report() {  # control first; the toy report compares with it
  CUDA_VISIBLE_DEVICES=0 $CL report --dir "$BASE/control" --device cuda > "$BASE/logs/report_control.txt" 2>&1
  CUDA_VISIBLE_DEVICES=0 $CL report --dir "$BASE/toy" --device cuda --control "$BASE/control" > "$BASE/logs/report_toy.txt" 2>&1
  tail -3 "$BASE/logs/report_control.txt" "$BASE/logs/report_toy.txt" >> "$BASE/logs/progress.log"
}
TOTAL=$(( $(ls "$BASE"/toy/fits/fit_*.toml | wc -l) + $(ls "$BASE"/control/fits/fit_*.toml | wc -l) ))

# 2. interim report
if [ "$(nseal)" -gt 0 ] && [ "$(nseal)" -lt "$TOTAL" ]; then log "phase 2: interim report ($(nseal)/$TOTAL sealed)"; report; fi

# 3. fits
if [ "$(nseal)" -lt "$TOTAL" ]; then
  # stale claims of earlier allocations are removed once here, never by the workers (race)
  $CL unlock-stale --dir "$BASE/toy" --dir "$BASE/control" 2>&1 | tail -1 | tee -a "$BASE/logs/progress.log"
  log "phase 3: fits, 4 GPUs x ${PER} workers ($(nseal)/$TOTAL sealed)"
  for g in 0 1 2 3; do
    for p in $(seq 1 "$PER"); do
      CUDA_VISIBLE_DEVICES=$g $CL fit --dir "$BASE/toy" --dir "$BASE/control" --claim \
        > "$BASE/logs/worker_${SLURM_JOB_ID:-local}_g${g}_$p.log" 2>&1 &
    done
  done
  wait
  log "phase 3 ended: $(nseal)/$TOTAL sealed"
fi

# 4. final
if [ "$(nseal)" -ge "$TOTAL" ]; then
  log "phase 4: final reports"; report
  grep -q "refused \[\]" "$BASE/logs/report_toy.txt" && touch "$BASE/DONE" && log "campaign DONE"
fi
