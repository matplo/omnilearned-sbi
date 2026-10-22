#!/bin/bash
# Export a commit of this repository as a read-only source snapshot for batch jobs:
#   shell/nle/snapshot.sh [REV]  -> prints $JEST_NLE_ROOT/src_snapshots/<rev>
# Jobs run with PYTHONPATH=<snapshot>/src, which takes precedence over the editable
# install, so later edits or branch switches in the checkout cannot reach them.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
REV=$(git -C "$REPO" rev-parse "${1:-HEAD}")
DEST=$JEST_NLE_ROOT/src_snapshots/$REV
if [ ! -f "$DEST/SOURCE_REVISION" ]; then
  mkdir -p "$DEST"
  git -C "$REPO" archive "$REV" | tar -x -C "$DEST"
  echo "$REV" > "$DEST/SOURCE_REVISION"
  chmod -R a-w "$DEST"
fi
echo "$DEST"
