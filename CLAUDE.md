# CLAUDE.md

Read `AGENTS.md` first. It is the source of truth for purpose, stack,
workflow, and data rules. This file only adds Claude Code specifics.

## Coding guidelines (always apply)

Adapted from [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)
(MIT per its README). Bias toward caution over speed; use judgment on
trivial tasks.

1. **Think before coding.** State assumptions; if uncertain, ask. If there
   are multiple interpretations, present them - don't pick silently. If a
   simpler approach exists, say so and push back. If something is unclear,
   stop and name what is confusing.
2. **Simplicity first.** Minimum code that solves the problem. No features
   beyond what was asked, no abstractions for single-use code, no
   unrequested configurability, no error handling for impossible cases. If
   200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what the task needs. Don't "improve"
   adjacent code, comments, or formatting; don't refactor what isn't
   broken; match existing style. Mention unrelated dead code, don't delete
   it. Remove only the imports/variables/functions your own change orphaned.
   Every changed line must trace to the task.
4. **Goal-driven execution.** Turn tasks into verifiable goals ("add
   validation" -> tests for invalid input, then make them pass; "fix bug" ->
   a failing reproduction test first). For multi-step work, state
   `step -> verify: check` before starting.

## Planning (OpenSpec vendor commands)

- `/opsx:explore` - think before committing to a change
- `/opsx:propose` - create a change; schema defaults to `rag-standard`,
  pass `--schema rag-rapid` only when AGENTS.md's tier rule allows it
- `/opsx:archive` - after merge-ready evidence exists

## Loops (project skills)

GitHub issues are the source of truth (see AGENTS.md Workflow). `gh`
must be signed in to the repo owner's account.

| Command | Loop | Edits code? | Does |
| --- | --- | --- | --- |
| `/rag-triage <issue#>` | triage | No | Preflight the issue, explore, propose, until `tasks.md` is complete; label it ready |
| `/rag-quick <issue#>` | developer | Via stages | Preflight (branch, labels), then plan -> execute -> commit per task, verify, review, PR |
| `/rag-plan <change>` | developer | No | Picks the next unchecked task, writes a short plan, waits for approval |
| `/rag-execute <change>` | developer | Yes | Implements exactly that task, self-reviews, ticks the checkbox |
| `/rag-review [PR#\|branch]` | developer | No | Reviews the diff against REVIEW.md, scope, complexity and prose; one line per finding; APPROVE or CHANGES |
| `/rag-verify <change>` | developer | Evidence only | Runs all checks + eval, writes `verification.md` |
| `/rag-pr <issue#>` | developer | PR only | Drafts the PR from the template; creates a **draft** PR on approval |

`/rag-quick` chains plan -> execute once per checkbox, then verify, review and PR;
the stage skills also work on their own.
`/opsx:apply` still works for `rag-rapid` changes, where the loop is overkill.

## Reviewing

Before reviewing any diff (including with a code-review skill), read
[REVIEW.md](REVIEW.md) and check the sections for the areas the diff
touches. `/rag-review` does this and adds scope, complexity and prose checks.

## Guardrails

`.claude/hooks/block-dangerous-git.mjs` denies force-push, `--no-verify`,
`reset --hard`, `clean -f`, `branch -D`, and asks before `rm -rf`.
