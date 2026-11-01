#!/bin/bash
# index + merge_v3 one JEWEL v3 split, on a compute node.
#
#   sbatch -A alice -C cpu -q debug -t 00:30:00 -N 1 shell/nle/jewel_v3_merge.sbatch \
#     SNAPSHOT TASK_DIR OUT_DIR SPLIT CONFIG
#
# Why not run this inline on the login node (jewel_v3_drive.sh did, until this script):
# merge_v3 loads every shard of a group fully into memory before writing (np.concatenate
# over all parts), and a login node's per-user memory limit is far below a compute node's.
# This OOM-killed the validation merge (1392 shards, 649 789 jets) with no traceback -- just
# "Killed" from the shell wrapping henv -- while the much smaller test merge (62 shards,
# 142 646 jets) had happened to fit. train and fidelity are both larger than validation.
#SBATCH -J v3_merge
#SBATCH -o %x-%j.log
set -euo pipefail
SNAP=$1; DIR=$2; OUT=$3; SPLIT=$4; CFG=$5
export PYTHONPATH=$SNAP/src
RUN="henv -n omni-sbi -x python"
$RUN -m omnilearned.sbi.jewel.generate index --dir "$DIR"
$RUN - "$DIR" "$OUT" "$SPLIT" "$CFG" <<'PYEOF'
import sys
from omnilearned.sbi.jewel import cards
from omnilearned.sbi.jewel.v3format import merge_v3
d, out, split, cfgp = sys.argv[1:5]
merge_v3(d, out, split, target_jets=1_000_000, cfg=cards.load_config(cfgp))
PYEOF
