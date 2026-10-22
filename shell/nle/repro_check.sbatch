#!/bin/bash
# M1 acceptance: regenerate shards on another node type and compare file hashes with the
# manifests; run the cross-host toy tests there.
#   sbatch -A alice -C cpu -q debug -t 00:15:00 -N 1 shell/nle/repro_check.sbatch SNAPSHOT DATA_DIR OUT_DIR "train:5 validation:3"
#SBATCH -J nle-repro
#SBATCH -o %x-%j.out
set -euo pipefail
SNAP=$1; DATA=$2; OUT=$3; SHARDS=$4
export PYTHONPATH=$SNAP/src OMP_NUM_THREADS=1
export CLING_STANDARD_PCH=${TMPDIR:-/tmp}/cling-$USER-$SLURM_JOB_ID/allDict.cxx.pch
mkdir -p "$(dirname "$CLING_STANDARD_PCH")" "$OUT"
PY=${PY:-$VIRTUAL_ENV/bin/python}
echo "job $SLURM_JOB_ID node $(hostname) $(grep -m1 'model name' /proc/cpuinfo) snapshot $(cat "$SNAP/SOURCE_REVISION")" | tee "$OUT/repro.log"
for spec in $SHARDS; do
  split=${spec%%:*}; k=${spec##*:}
  "$PY" -m omnilearned.sbi.toy.datasets generate --design "$DATA/design_$split.json" --shard "$k" \
    --points-per-shard 16 --jets $([ "$split" = train ] && echo 2000 || echo 1000) --out-dir "$OUT" >> "$OUT/repro.log" 2>&1
  f=$(printf "%s_%05d" "$split" "$k")
  "$PY" - "$DATA/$f.json" "$OUT/$f.json" <<'PYEOF' | tee -a "$OUT/repro.log"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = a["sha256"] == b["sha256"]
disc = all(x["discrete_sha256"] == y["discrete_sha256"] for x, y in zip(a["points"], b["points"]))
bit = all(x["array_sha256"] == y["array_sha256"] for x, y in zip(a["points"], b["points"]))
print(f"{a['file']}: file sha256 equal={same} discrete equal={disc} bitwise per point equal={bit} "
      f"host {a['host']['host']} -> {b['host']['host']}")
PYEOF
done
cd "$SNAP" && "$PY" -m pytest -q -p no:cacheprovider tests/sbi/test_toy.py::test_cross_host_reproducibility \
  tests/sbi/test_variable_multiplicity.py::test_v2_cross_host_reproducibility -rA 2>&1 | grep -E "PASSED|FAILED|SKIPPED|passed|failed" | tee -a "$OUT/repro.log"
