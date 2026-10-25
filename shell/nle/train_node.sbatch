#!/bin/bash
# Up to 4 surrogate trainings on one 4-GPU node (one per GPU) from a source snapshot, then the
# p(N|theta) of MULT_FROM attached to each flow (RUN/checkpoint_mult.pt), so that runs differ
# only in the flow. The job fails if any run fails. SPEC has one line per GPU: RUN_DIR [toy.train args...]; '#' lines skipped.
#   MULT_FROM=runs/m3/mult/set_d256_L6_200k_multC64.pt \
#   sbatch -A alice_g -C gpu -q regular -t 06:00:00 -N 1 --gpus 4 \
#     shell/nle/train_node.sbatch SNAPSHOT_DIR SPEC
#SBATCH -J nle-train4
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; SPEC=$2
export PYTHONPATH=$SNAP/src
PY=${PY:-henv -n omni-sbi -x python}  # always through henv (user rule)
$PY -c "import omnilearned, sys; assert omnilearned.__file__.startswith('$SNAP'), omnilearned.__file__"
k=0; pids=()
while read -r RUN ARGS; do
  case "$RUN" in ''|'#'*) continue ;; esac
  [ $k -lt 4 ] || { echo "more than 4 runs in $SPEC"; exit 1; }
  mkdir -p "$RUN"
  ( export CUDA_VISIBLE_DEVICES=$k
    echo "job $SLURM_JOB_ID node $(hostname) gpu $k snapshot $(cat "$SNAP/SOURCE_REVISION") start $(date -Is)" >> "$RUN/job.log"
    nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader -i $k >> "$RUN/job.log" || true
    # shellcheck disable=SC2086
    $PY -m omnilearned.sbi.toy.train --out "$RUN" $ARGS > "$RUN/stdout.log" 2>&1
    if [ -n "${MULT_FROM:-}" ]; then
      $PY -m omnilearned.sbi.toy.multiplicity_check attach --checkpoint "$RUN/checkpoint.pt" \
        --multiplicity-from "$MULT_FROM" --out "$RUN/checkpoint_mult.pt" >> "$RUN/job.log" 2>&1
    fi
    echo "end $(date -Is)" >> "$RUN/job.log" ) &
  pids+=($!); k=$((k + 1))
done < "$SPEC"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "$fail of $k runs failed"
[ $fail = 0 ]  # a failed run fails the job, so afterok dependents do not start
