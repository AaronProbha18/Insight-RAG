---
name: rag-pr
description: Prepare a draft pull request for an issue's branch from the PR template, the issue, verification.md and the diff; create it with gh only after approval. Use at the end of /rag-quick, or when the user says "prepare the PR for issue N".
allowed-tools: Bash(gh:*), Bash(git:*), Bash(openspec:*), Read, Grep, Glob, Write
---

Draft the PR for ONE issue. The issue stays the source of truth: the PR
closes it and says how each acceptance criterion was met.

**Input**: an issue number, e.g. `/rag-pr 42`.

## Steps

1. **Gather**
   - `gh issue view <N> --json number,title,body,url`.
   - The change linked by `Issue: #<N>` and its `verification.md`. For a
     rag-standard change, stop if the verdict isn't PASS. For a chore,
     record the checks that were run in this session.
   - `git log --oneline origin/main..HEAD` and `git diff --stat origin/main...HEAD`.
2. **Draft** the body from `.github/pull_request_template.md`:
   - `Closes #<N>` and the change path.
   - "What and why": 2-4 sentences, in terms of the issue's outcome.
   - "Verification": commands and results copied from `verification.md`.
     Never claim a check that wasn't run.
   - "Review": the `rag-review` verdict, the finding count and how many
     were fixed, or "not run".
   - For each acceptance criterion in the issue: the test that covers it,
     or "not covered" and why.
   - Tick only checklist items that are actually true; leave the rest
     unticked with a note.
   - Title: `<type>: <issue title> (#<N>)`.
3. **Show** the title and body to the user and wait for approval or edits.
4. **Create** on approval:
   `git push -u origin HEAD`, then
   `gh pr create --draft --title "<title>" --body-file <tmpfile> --base main
   --label <labels>`, copying the issue's `type:*`, `area:*` and
   `answer-quality` labels (never `status:*`, which tracks the issue, not
   the PR). Report the PR URL. Then `bash scripts/board.sh <N> "In Review"`.
   - `answer-quality` PRs: the body's eval line must hold real results. If
     it says "n/a", stop and say why that's wrong.
5. Suggest commenting the PR link on the issue. Do it only on approval.

## Guardrails

- Draft PRs only. Never mark one ready, merge, or close the issue.
- Never pushes or posts without approval in this session.
- No secrets, real personal data, internal URLs or org names in the title,
  body or commits.
