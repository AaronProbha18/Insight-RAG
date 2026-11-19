---
name: rag-plan
description: Plan the next unchecked tasks.md checkbox of an OpenSpec change as a short file-level micro-plan, then wait for approval. Use when starting work on a task in an active change, or when the user says "plan the next task".
allowed-tools: Bash(openspec:*), Read, Grep, Glob, Write
---

Produce a bounded plan for ONE `tasks.md` checkbox. Never edit code.

**Input**: `<change-name>` and optionally a task number (e.g. `2.1`). If no
change is given, run `openspec list --json`; if exactly one is active use
it, otherwise ask.

## Steps

1. `openspec status --change "<name>" --json`. If `tasks` is not done,
   stop: the change is not ready for implementation.
2. Read `proposal.md`, `design.md`, `tasks.md`, the relevant
   `specs/**/spec.md`, and `eval-plan.md` if present. Read any ADR that
   `design.md` links.
3. Pick the task: the one given, else the first unchecked `- [ ]`.
4. Read the code the task touches. Look for existing helpers, types, and
   patterns to reuse before planning anything new. **Read the matching
   guide before planning**, one per area the task touches, and follow its
   rules in the plan:
   - `frontend/` → `docs/guides/frontend-development-guide.md`
   - `backend/` → `docs/guides/backend-development-guide.md`
   - a connector, or `connectors/` → `docs/guides/connector-development-guide.md`

   Also read `docs/architecture/overview.md` when the task crosses
   modules. Name the guides you read in the plan's Scope section.
5. Write `openspec/changes/<name>/plan.md` (overwrite; one task at a time):

   ```markdown
   ## Task
   <number and text, verbatim>

   ## Scope
   - <what this loop does>
   ## Not in scope
   - <adjacent work deliberately left out>

   ## Files
   | File | Change |
   | --- | --- |

   ## Tests
   - <test that proves it, mapped to the spec scenario it covers>

   ## Checks to run
   - <exact commands>

   ## Risks / Questions
   - <anything unclear, or "none">
   ```

6. Show the plan and ask for approval. Do not proceed to execution.

## Stop and ask instead of planning when

- the task needs a decision `design.md` does not make, or contradicts an
  Accepted ADR,
- the task is too large for one loop (suggest how to split it in
  `tasks.md`),
- it needs a new external dependency or service not named in `design.md`.

## Guardrails

- Only `plan.md` is written. No code, no other artifacts.
- Never invent library or vendor APIs. Mark them UNVERIFIED with how to
  confirm them.
