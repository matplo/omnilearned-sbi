#!/bin/bash
# JEWEL v2 surrogate training: one node, 4 A100, 4 runs in parallel (one per GPU).
#
#   shell/nle/jewel_v2_train.sh SNAPSHOT OUTROOT [STEPS]
#   (inside an allocation: salloc -A alice_g -C gpu -q interactive -N 1 --gpus 4 -t 04:00:00 srun ...)
#
# Runs, chosen so the one question that matters is answered by the run set itself:
#   gpu 0  set 400k seed 0   \
#   gpu 1  set 400k seed 1    >  seed spread at the long schedule
#   gpu 2  set 400k seed 2   /
#   gpu 3  set 200k seed 0      <- 200k vs 400k at fixed seed
# Training length was the single lever that moved toy fidelity (400k beat 200k across 4
# seeds, while 2x data, weighting, d384 L8 and PET all landed inside the seed spread).
# That conclusion was never verified for JEWEL, so the 200k run is here to test it rather
# than assume it transfers, and the three seeds give the spread any comparison needs.
#
# Weighted training: the shards carry per-jet weights (WEXPO 5), and --weight-mode loss beat
# sample mode on the toy. p(N|theta) is retrained on JEWEL counts and attached to every run.
#
# NOTE ON VALIDATION LOSS: it is NOT comparable across differently-biased training splits --
# the feature transform is refitted per run on its own split. Judge fidelity with the gate
# and toy.ptbins, never with val loss. (This error was made twice: toy and JEWEL.)
set -uo pipefail
SNAP=$1; OUT=$2; STEPS=${3:-400000}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
DATASET=${DATASET:-jewel_v2k3}   # data/<DATASET>/merged; k3 = WEXPO 3 (log-flat pT-hat)
M=$JEST_NLE_ROOT/data/$DATASET/merged
CFG=$SNAP/configs/jewel-2.6.0-cs-v2.toml
PY="henv -n omni-sbi -x python"
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT/mult"
log() { echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] $*" | tee -a "$OUT/chunk.log"; }
$PY -c "import omnilearned,sys; sys.exit(0 if omnilearned.__file__.startswith('$SNAP') else 1)" \
  || { log "FATAL: not importing from the snapshot $SNAP"; exit 1; }
[ -f "$M/index_train.json" ] || { log "FATAL: no $M/index_train.json -- run Phase 2 first"; exit 1; }
log "start, snapshot $(cat "$SNAP/SOURCE_REVISION"), $STEPS steps, dataset $DATASET, data $M"
nvidia-smi -L | tee -a "$OUT/chunk.log"

COMMON=(--data-dir "$M" --val-dir "$M" --config "$CFG" --arch set --dim 256 --depth 6
        --heads 8 --batch-size 512 --lr 3e-4 --warmup-steps 1000 --ema-decay 0.9995
        --val-every 2000 --val-batches 16 --length-buckets 16 --tf32 --mult-steps 4000
        --weight-mode loss --save-after 10000 --device cuda --cpu-threads 8)
# Names carry the ACTUAL step count: this phase exists to compare training lengths, so a
# directory labelled 400k that holds a 200k run would invite exactly the wrong conclusion.
K=$((STEPS / 1000))
CTRL_STEPS=$(( STEPS > 200000 ? 200000 : STEPS / 2 ))
RUNS=(set_d256_L6_${K}k_s0 set_d256_L6_${K}k_s1 set_d256_L6_${K}k_s2 \
      set_d256_L6_$((CTRL_STEPS / 1000))k_ctl)
EXTRA=("--seed 0 --steps $STEPS" "--seed 1 --steps $STEPS" "--seed 2 --steps $STEPS"
       "--seed 0 --steps $CTRL_STEPS")
PIDS=()
for g in 0 1 2 3; do
  if [ -f "$OUT/${RUNS[$g]}/run.json" ]; then log "gpu $g: ${RUNS[$g]} already done, skip"; PIDS+=(0); continue; fi
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g $PY -m omnilearned.sbi.toy.train --out "$OUT/${RUNS[$g]}" \
    "${COMMON[@]}" ${EXTRA[$g]} > "$OUT/${RUNS[$g]}.out" 2>&1 &
  PIDS+=($!)
  log "gpu $g: ${RUNS[$g]} pid ${PIDS[$g]}"
done
for g in 0 1 2 3; do [ "${PIDS[$g]}" != 0 ] && { wait "${PIDS[$g]}"; log "${RUNS[$g]} ended rc=$?"; }; done

MULT=$OUT/mult/jewel_v2_multJ.pt
if [ ! -f "$MULT" ] && [ -f "$OUT/${RUNS[0]}/run.json" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY -m omnilearned.sbi.toy.multiplicity_check retrain \
    --checkpoint "$OUT/${RUNS[0]}/checkpoint.pt" --train-dir "$M" --val-dir "$M" \
    --out "$MULT" --device cuda >> "$OUT/mult/retrain.log" 2>&1
  log "p(N|theta) retrain rc=$?"
fi
for r in "${RUNS[@]}"; do
  [ -f "$OUT/$r/run.json" ] && [ -f "$MULT" ] || continue
  [ -f "$OUT/mult/${r}_multJ.pt" ] && continue
  $PY -m omnilearned.sbi.toy.multiplicity_check attach --checkpoint "$OUT/$r/checkpoint.pt" \
    --multiplicity-from "$MULT" --out "$OUT/mult/${r}_multJ.pt" >> "$OUT/mult/attach.log" 2>&1
  log "attached p(N|theta) to $r rc=$?"
done
[ -f "$MULT" ] && $PY -m omnilearned.sbi.toy.multiplicity_check check --checkpoint "$MULT" \
  --data-dir "$M" --split validation --out "$OUT/mult/multJ_check_validation.json" \
  >> "$OUT/mult/check.log" 2>&1
log "end"
