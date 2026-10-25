#!/bin/bash
# Toy-v2 shard generation as a Slurm job array (one shard per task, CPU node).
#   sbatch -A alice -C cpu -q regular -t 00:30:00 --array=0-63 \
#     shell/nle/gen_toy_array.sbatch DESIGN.json POINTS_PER_SHARD JETS OUT_DIR
# Several tasks share a node with --ntasks=1 --cpus-per-task=2 (the toy is single threaded).
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J toy-v2-gen
#SBATCH -o %x-%A_%a.out
set -euo pipefail
DESIGN=$1; PPS=$2; JETS=$3; OUT=$4
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")"
$PY -m omnilearned.sbi.toy.datasets generate --design "$DESIGN" --shard "$SLURM_ARRAY_TASK_ID" \
  --points-per-shard "$PPS" --jets "$JETS" --out-dir "$OUT" --skip-existing
