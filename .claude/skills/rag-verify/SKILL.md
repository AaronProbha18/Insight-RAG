---
name: rag-verify
description: Run the full check suite and the change's eval, then write verification.md with evidence and a PASS/FAIL verdict. Use when every tasks.md checkbox of an OpenSpec change is done, or the user says "verify the change".
allowed-tools: Bash, Read, Grep, Glob, Write
---

Produce evidence, not opinions. Never edit code.

**Input**: `<change-name>`. Same resolution rule as rag-plan.

## Steps

1. `openspec status --change "<name>" --json` and read `tasks.md`. If any
   checkbox is `[ ]`, stop and list them.
2. `openspec instructions verification --change "<name>" --json` for the
   template and instruction.
3. Run the project's full check suite (lint, typecheck, tests - the
   commands listed in `AGENTS.md`), then `openspec validate "<name>"`.
   If the change adds an Alembic revision, also run the drift check
   (`datamodel/README.md`, "Plumbline's model and the drift rule"). If it
   can't run locally, record it as pending CI, never as passed.
4. If `eval-plan.md` applies, run its "How to Run" command and compare
   each metric with the baseline and pass rule.
5. Map every `#### Scenario:` in the change's specs to the test that
   covers it. An uncovered scenario is a gap - list it.
6. Write `verification.md` from the template: every command with its
   result, scenario coverage, eval results, deviations from
   design/specs, and the Verdict.
7. Report the verdict. On PASS, the next steps are PR, human review,
   then `/opsx:archive`. On FAIL, list what must be fixed and which
   task it belongs to.

## Verdict rules

- FAIL if any check fails, any eval metric breaks the pass rule, or a
  spec scenario has no test.
- Never report a check you did not run in this session.
- Do not soften a FAIL or re-run until a flaky check passes without
  saying so.

## Guardrails

- Writes only `verification.md`. Fixes go back through rag-plan and
  rag-execute.
