---
name: rag-quick
description: Developer loop. For an issue that triage made ready, run plan -> execute -> verify for every task of its OpenSpec change, then draft the PR. Use when the user says "work on issue N" or runs /rag-quick N.
allowed-tools: Bash(gh:*), Bash(git:*), Bash(openspec:*), Read, Grep, Glob, Skill
---

Take ONE triaged issue to a draft PR by chaining the existing skills. This
skill orchestrates; the work happens in `rag-plan`, `rag-execute`,
`rag-verify` and `rag-pr`, whose rules all still apply.

**Input**: an issue number, e.g. `/rag-quick 42`.

## Steps

1. **Preflight**
   - `gh auth status` is signed in; `git status --porcelain` is empty.
   - `gh issue view <N> --json number,title,labels,state,url`. The issue is
     open and has `status:ready-for-dev`. If not, stop and suggest
     `/rag-triage <N>`.
   - `gh pr list --search "<N> in:body" --state open`. If an open PR
     already references the issue, stop and show it.
   - Find the change: `grep -rl "Issue: #<N>" openspec/changes
     --include=proposal.md`, outside `archive/`. Exactly one means use it.
     None, plus a "chore" triage comment on the issue, means the chore
     path (below). Anything else: stop and ask.
   - Sync and branch: `git fetch origin`, then
     `git switch -c <feat|fix|chore>/<N>-<slug> origin/main`, with the type
     from the issue's `type:*` label. If the branch exists, switch to it
     and continue where the tasks left off.
   - Tell the user, then swap `status:ready-for-dev` for
     `status:in-progress` (`gh issue edit <N> --remove-label
     status:ready-for-dev --add-label status:in-progress`), and
     `bash scripts/board.sh <N> "In Progress"`.
   - The issue has `answer-quality-change`? Then this run must end with an eval
     result in `verification.md`. Say so now, so the model setup is ready.
   - The issue has `db-change`? Then this run must end with the drift check
     recorded in `verification.md`. Say so now.

2. **Loop, once per unchecked `tasks.md` checkbox, in order**
   1. `rag-plan <change>`, then **wait for the user's approval** of
      `plan.md`.
   2. `rag-execute <change>`. If it pauses or stops, surface why and wait.
      Never skip a failed check to keep the loop moving.
   3. Commit that task on the branch:
      `<type>(<area>): <task summary> (#<N>)`. One commit per task.
   4. Report progress (`k/n tasks`) and continue to the next box.

   **Chore path** (no change): plan in the conversation, get approval,
   implement, run the AGENTS.md commands for the touched areas, commit.

3. **Verify**: `rag-verify <change>`. On FAIL, list the fixes and the tasks
   they belong to, and stop: fixes go back through the loop.

4. **PR**: `rag-pr <N>`.

## Stop and ask

- A task needs a decision the design doesn't make, or the issue's scope
  turns out wrong. Update the issue (with approval) rather than drift.
  While waiting, label the issue `status:blocked` and comment what it's
  waiting for; remove the label when work resumes.
- Any blocker you can't resolve inside the current task's scope.

## Guardrails

- One task at a time; every plan approved before execution.
- Never push, open a non-draft PR, merge or close the issue. `rag-pr`
  creates a **draft** PR, and only on approval.
