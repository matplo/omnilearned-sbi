#!/bin/bash
# Closure fits on the 4 GPUs of an allocated node (interactive job), PER_GPU processes each.
#   shell/nle/closure_local.sh SNAPSHOT CAMPAIGN_DIR [PER_GPU]
set -euo pipefail
SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 exec bash "$(dirname "$0")/closure_node.sbatch" "$@"
