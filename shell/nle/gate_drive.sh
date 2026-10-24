#!/bin/bash
# Drive gate_chunk.sh through back-to-back 4 h interactive GPU allocations (login node; screen
# or nohup) until both gates have all 128 points, MAX_CHUNKS allocations, or a chunk without
# progress.   nohup shell/nle/gate_drive.sh LOGDIR MAX_CHUNKS SNAPSHOT CKPT_A OUT_A CKPT_B OUT_B &
set -u
LOG=$1; MAX=$2; shift 2
SNAP=$1; OA=$3; OB=$5
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LOG"
n() { echo $(( $(ls "$OA"/gate/point_*.json 2>/dev/null | wc -l) + $(ls "$OB"/gate/point_*.json 2>/dev/null | wc -l) )); }
log() { echo "[$(date -Is) driver $(hostname)] $*" | tee -a "$LOG/progress.log"; }
for i in $(seq 1 "$MAX"); do
  [ "$(n)" -ge 256 ] && { log "both gates complete"; break; }
  [ -f "$LOG/STOP" ] && { log "STOP file found"; break; }
  before=$(n)
  log "chunk $i/$MAX: requesting a 4 h interactive GPU node ($before/256 gate points)"
  salloc -A alice_g -C gpu -q interactive -N 1 -n 1 -c 128 -t 04:00:00 --gpus 4 -J m3-gate \
    srun -N 1 -n 1 -c 128 --gpus 4 --cpu-bind=none bash "$HERE/gate_chunk.sh" "$@" > "$LOG/chunk_$i.log" 2>&1
  after=$(n)
  log "chunk $i ended rc=$?, gate points $before -> $after"
  [ "$after" -le "$before" ] && { log "no progress: stopping (see chunk log)"; break; }
  sleep 60
done
