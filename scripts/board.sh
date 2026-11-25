#!/usr/bin/env bash
# Set an issue's Status on the project board, adding it first if missing.
# Usage: scripts/board.sh <issue#> <Todo|Ready|In Progress|In Review|Done>
# Needs: gh auth refresh -s project
set -euo pipefail
N=$1; STATUS=$2
OWNER=AaronProbha18; NUM=1
PROJECT=PVT_kwHOCTs1n84Bs_K7
FIELD=PVTSSF_lAHOCTs1n84Bs_K7zhrKtDA
case $STATUS in
  Todo) OPT=9a948480 ;; Ready) OPT=417948d0 ;; "In Progress") OPT=559c1d1e ;;
  "In Review") OPT=a587441e ;; Done) OPT=37671ecd ;;
  *) echo "unknown status: $STATUS" >&2; exit 1 ;;
esac
URL=$(gh issue view "$N" --json url -q .url)
ITEM=$(gh project item-add $NUM --owner $OWNER --url "$URL" --format json -q .id)
gh project item-edit --project-id $PROJECT --id "$ITEM" --field-id $FIELD --single-select-option-id $OPT >/dev/null
echo "#$N -> $STATUS"
