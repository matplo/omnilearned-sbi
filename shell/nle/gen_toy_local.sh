#!/bin/bash
# Toy-v2 shard generation on the CPUs of an allocated node (e.g. an interactive job).
#   shell/nle/gen_toy_local.sh DESIGN.json NSHARDS POINTS_PER_SHARD JETS OUT_DIR [PARALLEL] [extra generate args]
# Resumable: shards with a manifest are skipped. Logs in OUT_DIR/logs/.
set -euo pipefail
DESIGN=$1; NSH=$2; PPS=$3; JETS=$4; OUT=$5; PAR=${6:-48}; shift $(( $# < 6 ? $# : 6 )); EXTRA="$*"
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule); unquoted below (several words)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT/logs"
$PY -c "from omnilearned.sbi.toy import reco; reco.fastjet_version()" >/dev/null 2>&1  # build the PCH once
seq 0 $((NSH - 1)) | xargs -P "$PAR" -I{} sh -c \
  "$PY -m omnilearned.sbi.toy.datasets generate --design \"$DESIGN\" --shard {} \
   --points-per-shard $PPS --jets $JETS --out-dir \"$OUT\" --skip-existing $EXTRA > \"$OUT/logs/shard_{}.log\" 2>&1 \
   || echo \"shard {} FAILED\""
