#!/bin/bash
# Phase 4: fidelity of a JEWEL v2 surrogate -- the 128-point gate plus per-pT bins.
#   shell/nle/jewel_v2_gate.sh SNAPSHOT CKPT OUTROOT [DATASET]
# One node, 4 GPUs: the gate is sharded 4 ways, then ptbins runs on 4 GPUs by point.
# Both tools take --config so the JEWEL selection (20-500 GeV, |eta|<2, inclusive) is used;
# ptbins derives its bin edges from that window rather than the toy's 60-200 default.
set -uo pipefail
SNAP=$1; CKPT=$2; OUT=$3; DATASET=${4:-jewel_v2k3}
export JEST_NLE_ROOT=${JEST_NLE_ROOT:-/global/cfs/cdirs/alice/ploskon/jest-nle}
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-${SLURM_JOB_ID:-local}/allDict.cxx.pch
M=$JEST_NLE_ROOT/data/$DATASET/merged
CFG=$SNAP/configs/jewel-2.6.0-cs-v2.toml
PY="henv -n omni-sbi -x python"
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT/gate" "$OUT/ptbins"
log() { echo "[$(date -Is) $(hostname) job ${SLURM_JOB_ID:-local}] $*" | tee -a "$OUT/gate.log"; }
[ -f "$M/index_fidelity.json" ] || { log "FATAL: no $M/index_fidelity.json"; exit 1; }
[ -f "$CKPT" ] || { log "FATAL: no checkpoint $CKPT"; exit 1; }
log "start, snapshot $(cat "$SNAP/SOURCE_REVISION"), ckpt $(basename "$CKPT"), data $M"

log "gate: 128 fidelity points, 4 shards, redraw (per-N selection normalisation)"
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY -m omnilearned.sbi.toy.gate run --checkpoint "$CKPT" \
    --fidelity-dir "$M" --config "$CFG" --redraw --n-generate 22000 \
    --shard $g --num-shards 4 --device cuda --out "$OUT/gate" \
    > "$OUT/gate_shard$g.log" 2>&1 &
done
wait
log "gate shards done: $(ls "$OUT"/gate/point_*.json 2>/dev/null | wc -l)/128 points"
$PY -m omnilearned.sbi.toy.gate summarize --dir "$OUT/gate" --n-settings 128 \
  > "$OUT/gate/summary.txt" 2>&1
tail -1 "$OUT/gate/summary.txt" | tee -a "$OUT/gate.log"

log "ptbins: per-pT fidelity over the config window (20-500 GeV)"
PTS=(0 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120)
for g in 0 1 2 3; do
  sel=(); for i in "${!PTS[@]}"; do [ $((i % 4)) -eq $g ] && sel+=("${PTS[$i]}"); done
  CUDA_VISIBLE_DEVICES=$g $PY -m omnilearned.sbi.toy.ptbins run --checkpoint "$CKPT" \
    --fidelity-dir "$M" --config "$CFG" --points "${sel[@]}" --device cuda \
    --out "$OUT/ptbins" > "$OUT/ptbins_shard$g.log" 2>&1 &
done
wait
log "ptbins done: $(ls "$OUT"/ptbins/point_*.json 2>/dev/null | wc -l)/${#PTS[@]} points"
log "end"
