#!/bin/bash
# Drive a closure campaign through back-to-back 4 h interactive GPU allocations, from a login
# node (run inside screen/tmux, or detached with nohup). Stops when BASE/DONE exists, after
# MAX_CHUNKS allocations, or after two consecutive chunks without a new sealed fit.
#   nohup shell/nle/m5_drive.sh BASE [MAX_CHUNKS] > BASE/logs/drive.out 2>&1 &
set -u
BASE=$(cd "$1" && pwd); MAX=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$BASE/logs"
nseal() { ls "$BASE"/toy/fits/*/seal.json "$BASE"/control/fits/*/seal.json 2>/dev/null | wc -l; }
log() { echo "[$(date -Is) driver $(hostname)] $*" | tee -a "$BASE/logs/progress.log"; }
echo $$ > "$BASE/logs/driver.pid"
stall=0
for i in $(seq 1 "$MAX"); do
  [ -f "$BASE/DONE" ] && { log "DONE"; break; }
  [ -f "$BASE/STOP" ] && { log "STOP file found, not requesting more allocations"; break; }
  before=$(nseal)
  touch "$BASE/logs/.chunk_start"
  log "chunk $i/$MAX: requesting a 4 h interactive GPU node (sealed $before)"
  salloc -A alice_g -C gpu -q interactive -N 1 -n 1 -c 128 -t 04:00:00 --gpus 4 -J m5-closure \
    srun -N 1 -n 1 -c 128 --gpus 4 --cpu-bind=none bash "$HERE/m5_dev_closure.sh" "$BASE" \
    > "$BASE/logs/chunk_$i.log" 2>&1
  rc=$?
  after=$(nseal)
  # progress = new seals, or any acceptance part / fit cache written during the chunk
  touched=$(find "$BASE" -name "*.npz" -newer "$BASE/logs/.chunk_start" 2>/dev/null | wc -l)
  log "chunk $i ended rc=$rc, sealed $before -> $after, npz files written $touched"
  if [ "$after" -le "$before" ] && [ "$touched" -eq 0 ] && [ ! -f "$BASE/DONE" ]; then
    stall=$((stall + 1)); else stall=0; fi
  [ "$stall" -ge 2 ] && { log "two chunks without progress: stopping (see chunk logs)"; break; }
  sleep 60
done
rm -f "$BASE/logs/driver.pid"
