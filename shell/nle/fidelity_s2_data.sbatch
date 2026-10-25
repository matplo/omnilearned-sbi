#!/bin/bash
# Fidelity plan S2 training data on one CPU node (toy v2, backend unchanged):
#   v2_m3a: the round-2 design (1024 points) with 12000 jets per point (option a);
#   v2_m3b: the round-2 shards (links) plus the extra Sobol points 1024-2047 of the train
#           design with T_ref > 0.45 (343 points, density x2 there) at 6000 jets (option b),
#           as shards 64.. next to them.
# Both link the round-2 validation split. Refuses to touch a directory that has a train index.
#   sbatch -A alice -C cpu -q regular -t 00:45:00 -N 1 shell/nle/fidelity_s2_data.sbatch SNAPSHOT_DIR
#SBATCH -J nle-s2-data
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
M2=$JEST_NLE_ROOT/data/v2_m2; A=$JEST_NLE_ROOT/data/v2_m3a; B=$JEST_NLE_ROOT/data/v2_m3b
for d in "$A" "$B"; do
  [ ! -e "$d/index_train.json" ] || { echo "$d already has a train index"; exit 1; }
  mkdir -p "$d/logs"
  [ -e "$d/index_validation.json" ] || ln -s "$M2"/validation_* "$M2/index_validation.json" "$M2/design_validation.json" "$d/"
  echo "$(cat "$SNAP/SOURCE_REVISION") job $SLURM_JOB_ID $(date -Is)" > "$d/SOURCE_NOTE"
done
[ -e "$A/design_train.json" ] || cp "$M2/design_train.json" "$A/design_train.json"
[ -e "$B/train_00000.h5" ] || ln -s "$M2"/train_* "$B/"
[ -e "$B/design_train_ext.json" ] || $PY -m omnilearned.sbi.toy.datasets design --split train --n 2048 \
  --from-index 1024 --select T_ref 0.45 0.55 --out "$B/design_train_ext.json"
NEXT=$($PY -c "import json; print(len(json.load(open('$B/design_train_ext.json'))['points']))")
NB=$(( (NEXT + 15) / 16 ))
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1  # build the PCH once
echo "start $(date -Is): v2_m3a 64 shards x 16 points x 12000 jets, v2_m3b $NB shards ($NEXT points) x 6000 jets"
{ for k in $(seq 0 63); do echo "$A/design_train.json $k 12000 $A 0"; done
  for k in $(seq 0 $((NB - 1))); do echo "$B/design_train_ext.json $k 6000 $B 64"; done; } |
  xargs -P 120 -L 1 sh -c "$PY -m omnilearned.sbi.toy.datasets generate --design \$0 --shard \$1 \
    --points-per-shard 16 --jets \$2 --out-dir \$3 --shard-offset \$4 --skip-existing \
    > \$3/logs/shard_\$1.log 2>&1 || echo \"shard \$0 \$1 FAILED\""
$PY -m omnilearned.sbi.toy.datasets index --dir "$A" --split train
$PY -m omnilearned.sbi.toy.datasets index --dir "$B" --split train
echo "end $(date -Is)"
