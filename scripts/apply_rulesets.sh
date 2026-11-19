#!/bin/sh
# Apply .github/rulesets/*.json to the current repo, and set squash-only merging.
# Needs `gh` signed in as a repo admin. Safe to re-run: a ruleset with the same
# name is updated in place, otherwise created. Rulesets on GitHub that have no
# file here are left alone.
set -eu
cd "$(dirname "$0")/.."
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

# Repo-level merge settings: squash only, branches cleaned up after merge.
gh api -X PATCH "repos/$repo" \
  -F allow_squash_merge=true -F allow_merge_commit=false -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true \
  -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY >/dev/null
echo "merge settings: squash only"

for file in .github/rulesets/*.json; do
  name=$(sed -n 's/^  "name": "\([^"]*\)",$/\1/p' "$file")
  id=$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")
  if [ -n "$id" ]; then
    gh api -X PUT "repos/$repo/rulesets/$id" --input "$file" >/dev/null
    echo "ruleset '$name' updated (#$id)"
  else
    id=$(gh api -X POST "repos/$repo/rulesets" --input "$file" --jq .id)
    echo "ruleset '$name' created (#$id)"
  fi
done
