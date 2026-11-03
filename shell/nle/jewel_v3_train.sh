#!/bin/bash
# JEWEL v3 surrogate training: one node, 4 A100, 4 runs in parallel (one per GPU).
#
#   shell/nle/jewel_v3_train.sh SNAPSHOT OUTROOT [STEPS]
#   (inside an allocation: sbatch -A alice_g -C gpu -q regular -N 1 --gpus 4 -t 08:00:00 ...)
#
# Mirrors jewel_v2_train.sh's run set (plan rule: training length is the one lever that moved
# toy fidelity, never verified for JEWEL until measured directly):
#   gpu 0  set 400k seed 0   \
#   gpu 1  set 400k seed 1    >  seed spread at the long schedule
#   gpu 2  set 400k seed 2   /
#   gpu 3  set 200k seed 0      <- 200k vs 400k at fixed seed
#
# v3-specific, unlike v2: toy.train dispatches on --config's backend id ("-v3" suffix) to
# jewel.v3format's reader, conditions BOTH the flow and the per-run multiplicity model on
# jet_pt (never the multiplicity model on n_constituents -- see layout_for_multiplicity_model),
# and skips FeatureTransform fitting (identity transform; docs/FORMAT_JEWEL_V3.md: the
# paper's minimal 4-feature set is already O(1)-scaled). Each run's own checkpoint therefore
# already carries a jet_pt-conditioned p(N|pT_J,theta) -- no separate retrain+attach step
# (v2's shared-mult comparison technique does not apply here: nothing about the v3 plan
# needs runs to share one p(N|theta) estimate, and multiplicity_check.py is not v3-aware).
#
# NOTE ON VALIDATION LOSS: it is NOT comparable across differently-biased training splits --
# the feature transform is refitted per run on its own split (identity here, but still
# per-run). Judge fidelity with the gate and toy.ptbins, never with val loss.
set -uo pipefail
SNAP=$1; OUT=$2; STEPS=${3:-400000}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
DATASET=${DATASET:-jewel_v3}
M=$JEST_NLE_ROOT/data/$DATASET/merged
CFG=$SNAP/configs/jewel-2.6.0-cs-v3.toml
PY="henv -n omni-sbi -x python"
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT"
log() { echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] $*" | tee -a "$OUT/chunk.log"; }
$PY -c "import omnilearned,sys; sys.exit(0 if omnilearned.__file__.startswith('$SNAP') else 1)" \
  || { log "FATAL: not importing from the snapshot $SNAP"; exit 1; }
[ -f "$M/index_train.json" ] || { log "FATAL: no $M/index_train.json -- run P2 first"; exit 1; }
log "start, snapshot $(cat "$SNAP/SOURCE_REVISION"), $STEPS steps, dataset $DATASET, data $M"
nvidia-smi -L | tee -a "$OUT/chunk.log"

COMMON=(--data-dir "$M" --val-dir "$M" --config "$CFG" --arch set --dim 256 --depth 6
        --heads 8 --batch-size 512 --lr 3e-4 --warmup-steps 1000 --ema-decay 0.9995
        --val-every 2000 --val-batches 16 --length-buckets 16 --tf32 --mult-steps 4000
        --weight-mode loss --save-after 10000 --device cuda --cpu-threads 8)
K=$((STEPS / 1000))
CTRL_STEPS=$(( STEPS > 200000 ? 200000 : STEPS / 2 ))
RUNS=(v3_set_d256_L6_${K}k_s0 v3_set_d256_L6_${K}k_s1 v3_set_d256_L6_${K}k_s2 \
      v3_set_d256_L6_$((CTRL_STEPS / 1000))k_ctl)
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
fail=0
for g in 0 1 2 3; do
  [ "${PIDS[$g]}" != 0 ] && { wait "${PIDS[$g]}" || fail=$((fail + 1)); log "${RUNS[$g]} ended rc=$?"; }
done
log "end, $fail of 4 runs failed"
[ "$fail" = 0 ]
