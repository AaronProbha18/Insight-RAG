---
name: rag-triage
description: Triage loop. Turn a GitHub issue into a fully planned OpenSpec change (explore, propose, until tasks.md is complete), then hand it to development. Use when the user says "triage issue N" or runs /rag-triage N. Never edits code.
allowed-tools: Bash(gh:*), Bash(git:*), Bash(openspec:*), Read, Grep, Glob, Write, Edit, Skill
---

Turn ONE GitHub issue into a change a developer can run `/rag-quick` on.
The issue is the source of truth: if scope changes during triage, write
it back to the issue. Never edit code.

**Input**: an issue number, e.g. `/rag-triage 42`.

## Steps

1. **Preflight**
   - `gh auth status`. If not signed in to the repo's owner account, stop
     and ask the user to run `gh auth login`.
   - `git status --porcelain` is empty, and `git fetch origin` succeeds.
   - `gh issue view <N> --json number,title,body,labels,state,url,comments`.
     Stop if the issue is closed or doesn't exist.
   - The issue has a problem statement, acceptance criteria, and an
     "answer quality" answer (the issue forms ask for all three). If one is
     missing, draft a comment asking for it, show it to the user, post it
     only on approval (`gh issue comment`), and stop.
   - If an open change already names this issue (`grep -rl "Issue: #<N>"
     openspec/changes --include=proposal.md`), stop: triage is already
     done; suggest `/rag-quick <change>`.

2. **Route**
   - **Repo-internal** (CI, lint, `.claude/`, schemas, dependency bumps,
     docs only): no OpenSpec change (AGENTS.md rule 2). Post a comment
     saying it's a chore, swap `status:needs-triage` for
     `status:ready-for-dev`, add the `area:*` labels,
     and stop.
   - **Answer quality "Yes"** or it touches retrieval, prompts,
     connectors, ingestion, a public contract or the database: schema
     `rag-standard`.
   - Otherwise: `rag-rapid`. When unsure, use `rag-standard`.
   - Tell the user the route and why, and wait for approval.

3. **Explore**: invoke `opsx:explore` with the issue's title, problem,
   outcome and acceptance criteria. Read
   `docs/architecture/overview.md` and the guide for each ticked area
   first. Settle open questions with the user here, not in code.

4. **Propose**: invoke `opsx:propose <kebab-name>` (plus
   `--schema rag-rapid` if routed there). The proposal's first line is
   `Issue: #<N>`. Every acceptance criterion from the issue becomes a spec
   scenario (rag-standard) or a task (rag-rapid).
   **Database change:** if the change adds an Alembic revision, the same
   task in `tasks.md` also updates `datamodel/models/plumbline/` (ADR 0004).

5. **Until tasks.md is complete**: `openspec status --change <name>`
   shows every artifact up to `tasks` done, and `openspec validate <name>`
   passes. Each task fits one plan/execute loop; split any that don't.

6. **Hand off** (show each to the user; act on approval):
   - Comment on the issue: the change path, the schema, the task count,
     and anything the issue's scope gained or lost during triage.
   - Labels (all defined in `.github/labels.yml`):
     `gh issue edit <N> --remove-label status:needs-triage --add-label
     status:ready-for-dev,area:<…>`, plus
     `answer-quality-change` if the form said "Yes" or the route is
     `rag-standard`, and `db-change` if the form said "Yes" or "Not sure",
     or the change adds an Alembic revision. If triage itself is stuck on a
     decision, add `status:blocked` and say what it's waiting for.
   - Board: `bash scripts/board.sh <N> Ready`.
   - Commit the change's planning artifacts on the issue's one branch,
     `<feat|fix|chore>/<N>-<slug>` (type from the `type:*` label, as in
     `rag-quick`), with message `docs(openspec): plan #<N> <title>`. Never
     a separate `triage/` branch: triage, plan, execute, verify, review and
     archive all happen on this branch, in one PR.
   - Push it and open a **draft** PR from it that `Closes #<N>`, titled
     `<type>: <issue title> (#<N>)`, with the issue's `type:*`, `area:*`,
     `answer-quality-change` and `db-change` labels. Triage ends here;
     execution comes later on the same branch.

7. **Report**: the change name, its tasks, the issue URL, the draft PR
   URL, and the next step: `/rag-quick <N>` (it finds the branch and
   continues on it).

## Guardrails

- Never edits code, and never starts `rag-plan` or `rag-execute`.
- Never posts to GitHub without showing the text first and getting
  approval.
- Synthetic data only. If the issue contains anything resembling real
  personal data or secrets, stop and ask the reporter to redact it.
