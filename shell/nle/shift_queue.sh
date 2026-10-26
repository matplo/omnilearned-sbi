#!/bin/bash
# Work queue of high-statistics equivalent shifts (gate shift --direct-dir) on one GPU. Each
# worker takes the next model whose output directory it can create (mkdir is the lock), so any
# number of workers (interactive GPUs, batch jobs) can share one queue file. The queue file is
# re-read after every model, so lines can be appended while workers run.
#   shell/nle/shift_queue.sh SNAPSHOT QUEUE_FILE GPU
# QUEUE_FILE lines: NAME CHECKPOINT REDRAW(0|1); '#' lines skipped.
# Output: $OUTROOT/NAME/{shift.json,shift.log,job.log}; a run that fails leaves NAME/FAILED.
# A directory with job.log but neither shift.json nor FAILED is in progress or was killed.
set -uo pipefail
SNAP=$1; Q=$2; GPU=$3
export PYTHONPATH=$SNAP/src CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
SCAN=${SCAN:-$JEST_NLE_ROOT/data/v2_m1_devscan}
DIRECT=${DIRECT:-$JEST_NLE_ROOT/data/v2_m3_shiftdirect}
OUTROOT=${OUTROOT:-$JEST_NLE_ROOT/runs/m3/shift_hi}
NGEN=${NGEN:-50000}
$PY -c "import omnilearned; assert omnilearned.__file__.startswith('$SNAP'), omnilearned.__file__" || exit 1
until [ -f "$DIRECT/index_development_scan.json" ]; do sleep 60; done  # direct sample still generating
mkdir -p "$OUTROOT"
while true; do
  NAME=""
  while read -r N C R; do
    case "$N" in ''|'#'*) continue ;; esac
    if mkdir "$OUTROOT/$N" 2>/dev/null; then NAME=$N; CKPT=$C; RD=$R; break; fi
  done < "$Q"
  [ -n "$NAME" ] || break
  OUT=$OUTROOT/$NAME
  echo "job ${SLURM_JOB_ID:-} node $(hostname) gpu $GPU snapshot $(cat "$SNAP/SOURCE_REVISION") ckpt $CKPT sha256 $(sha256sum "$CKPT" | cut -c1-16) redraw $RD n_generate $NGEN direct $DIRECT start $(date -Is)" >> "$OUT/job.log"
  ARGS=(); [ "$RD" = 1 ] && ARGS=(--redraw)
  if $PY -m omnilearned.sbi.toy.gate shift --checkpoint "$CKPT" --scan-dir "$SCAN" --direct-dir "$DIRECT" \
      --out "$OUT/shift.json" --n-generate "$NGEN" --device cuda "${ARGS[@]}" > "$OUT/shift.log" 2>&1; then
    echo "end $(date -Is)" >> "$OUT/job.log"
  else
    echo "FAILED $(date -Is)" | tee "$OUT/FAILED" >> "$OUT/job.log"
  fi
done
echo "queue $Q empty for gpu $GPU $(date -Is)"
