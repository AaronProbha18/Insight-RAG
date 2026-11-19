---
name: rag-execute
description: Implement exactly the approved plan.md task of an OpenSpec change, self-review the diff, run its checks, and tick the tasks.md checkbox. Use after rag-plan was approved, or when the user says "execute the plan".
---

Implement ONE planned task. Nothing beyond its stated scope.

**Input**: `<change-name>`. Same resolution rule as rag-plan.

## Steps

1. Read `openspec/changes/<name>/plan.md`. If it is missing, or its task
   is already `[x]` in `tasks.md`, stop and tell the user to run rag-plan.
2. Re-read `design.md` and the specs the plan references, and the dev
   guide for every area in the Files table (`frontend/` →
   `docs/guides/frontend-development-guide.md`, `backend/` →
   `docs/guides/backend-development-guide.md`, connectors →
   `docs/guides/connector-development-guide.md`). The code you write
   follows those rules; if a rule is wrong, stop and say so rather than
   break it.
3. Implement the plan's Files table. Write the tests from its Tests
   section alongside the code.
4. Run every command in "Checks to run". On failure: fix if the cause is
   inside this task's scope; otherwise pause and report.
5. Self-review the diff against the plan:
   - anything changed outside the Files table?
   - anything in "Not in scope" touched?
   - every changed line traces to the task (CLAUDE.md coding guidelines 2-3):
     no unrequested abstraction, config, or adjacent reformatting?
   - debug output, dead code, secrets, real data in fixtures?
   - unverified library behaviour asserted as fact?
   - docs: did this change behaviour, a contract, a boundary rule or a
     command that `docs/architecture/overview.md` or a guide in
     `docs/guides/` describes? Update them in this task, or say why not
     (AGENTS.md workflow rule 7).
   Fix what is in scope; report the rest.
6. Tick the task's checkbox in `tasks.md`.
7. Report: files changed, checks run with results, review notes, and the
   next unchecked task (suggest rag-plan for it, or rag-verify if none
   remain).

## Pause (in-session, the user resolves)

- a check fails for a reason outside this task,
- the self-review finds something you are unsure about.

## Stop and ask (needs a design decision)

- the plan turns out to need a design change or contradicts an Accepted ADR,
- a new external dependency or service is required,
- the task is larger than planned - propose a split in `tasks.md`.

## Guardrails

- One task per run. Do not start the next task.
- Never commit, push, or merge unless the user asks.
- Never mark a checkbox done when its checks did not pass.
