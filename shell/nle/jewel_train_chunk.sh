#!/bin/bash
# JEWEL surrogate trainings on one 4-GPU node, one training per GPU: set and smooth-PET flows,
# seeds 0 and 1 (the seed pair measures training-to-training spread), toy round-2 settings.
# When the seed-0 set flow is done, p(N|theta) is retrained on all JEWEL training counts on
# its GPU; at the end it is attached to every flow (<run>_multJ.pt) and checked against the
# validation split. Not resumable: a training cut by the limit keeps its best checkpoint so far
# (saved from step 10k on) but no run.json. The ~30 k per-task shards are slow to open on CFS
# (≈ 4 files/s per process), so the splits are first merged into data/jewel_v1/merged (64 + 8
# point-major shards, `generate merge`, skipped if present) and the trainings read those.
#   shell/nle/jewel_train_chunk.sh SNAPSHOT OUTROOT [STEPS]
set -uo pipefail
SNAP=$1; OUT=$2; STEPS=${3:-200000}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export VIRTUAL_ENV=${VIRTUAL_ENV:-/global/cfs/cdirs/alice/ploskon/henv-omni-sbi/.venv}
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER/allDict.cxx.pch PYTHONUNBUFFERED=1
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8
PY=$VIRTUAL_ENV/bin/python
D=$JEST_NLE_ROOT/data/jewel_v1
M=$D/merged
CFG=$SNAP/configs/jewel-2.6.0-cs-v1.toml
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT/mult"
"$PY" -c "import omnilearned; assert omnilearned.__file__.startswith('$SNAP'), omnilearned.__file__"
log() { echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] $*" | tee -a "$OUT/chunk.log"; }
log "start, snapshot $(cat "$SNAP/SOURCE_REVISION"), $STEPS steps"
nvidia-smi -L | tee -a "$OUT/chunk.log"
for sp in train:64 validation:8; do
  [ -f "$M/index_${sp%%:*}.json" ] && continue
  "$PY" -m omnilearned.sbi.jewel.generate merge --dir "$D/${sp%%:*}" --out "$M" --shards "${sp##*:}" \
    --workers 32 2>&1 | tee -a "$OUT/chunk.log"
  [ -f "$M/index_${sp%%:*}.json" ] || { log "merge of ${sp%%:*} failed"; exit 1; }
done

COMMON=(--data-dir "$M" --val-dir "$M" --config "$CFG" --dim 256 --depth 6
        --heads 8 --steps "$STEPS" --batch-size 512 --lr 3e-4 --warmup-steps 1000 --ema-decay 0.9995
        --val-every 2000 --val-batches 16 --length-buckets 16 --tf32 --mult-steps 4000
        --save-after 10000 --device cuda --cpu-threads 8)
RUNS=(set_d256_L6_s0 pet_d256_L6_s0 set_d256_L6_s1 pet_d256_L6_s1)
EXTRA=("--arch set --seed 0" "--arch pet_smooth --local-dim 64 --seed 0"
       "--arch set --seed 1" "--arch pet_smooth --local-dim 64 --seed 1")
PIDS=()
for g in 0 1 2 3; do
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g "$PY" -m omnilearned.sbi.toy.train --out "$OUT/${RUNS[$g]}" "${COMMON[@]}" \
    ${EXTRA[$g]} > "$OUT/${RUNS[$g]}.out" 2>&1 &
  PIDS+=($!)
  log "gpu $g: ${RUNS[$g]} pid ${PIDS[$g]}"
done

wait "${PIDS[0]}"; log "${RUNS[0]} ended rc=$?"
MULT=$OUT/mult/jewel_multJ.pt
if [ -f "$OUT/${RUNS[0]}/run.json" ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" -m omnilearned.sbi.toy.multiplicity_check retrain \
    --checkpoint "$OUT/${RUNS[0]}/checkpoint.pt" --train-dir "$M" --val-dir "$M" \
    --out "$MULT" --device cuda >> "$OUT/mult/retrain.log" 2>&1
  log "p(N|theta) retrain rc=$?"
fi
for g in 1 2 3; do wait "${PIDS[$g]}"; log "${RUNS[$g]} ended rc=$?"; done

for r in "${RUNS[@]}"; do
  [ -f "$OUT/$r/run.json" ] && [ -f "$MULT" ] || continue
  "$PY" -m omnilearned.sbi.toy.multiplicity_check attach --checkpoint "$OUT/$r/checkpoint.pt" \
    --multiplicity-from "$MULT" --out "$OUT/mult/${r}_multJ.pt" >> "$OUT/mult/attach.log" 2>&1
done
for c in "$OUT/${RUNS[0]}/checkpoint.pt" "$MULT"; do
  [ -f "$c" ] && "$PY" -m omnilearned.sbi.toy.multiplicity_check check --checkpoint "$c" \
    --data-dir "$M" --split validation --out "${c%.pt}_check_validation.json" \
    >> "$OUT/mult/check.log" 2>&1
done
log "end"
