#!/bin/sh
# Create or update every label in .github/labels.yml on the current repo's GitHub.
# Needs `gh` signed in with write access. Safe to re-run (--force updates in place).
# Labels on GitHub that are not in the file are left alone.
set -eu
cd "$(dirname "$0")/.."
sed -n 's/^- {name: "\([^"]*\)", color: "\([^"]*\)", description: "\([^"]*\)"}$/\1|\2|\3/p' .github/labels.yml |
while IFS='|' read -r name color description; do
  gh label create "$name" --color "$color" --description "$description" --force
done
echo "labels synced: $(grep -c '^- {name:' .github/labels.yml)"
