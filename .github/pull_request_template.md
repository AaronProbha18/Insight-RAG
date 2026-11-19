<!-- The linked issue is the source of truth. Draft this with /rag-pr. -->

Closes #<!-- issue number -->

**Change:** `openspec/changes/<name>/` (or "chore: no OpenSpec change", AGENTS.md rule 2)

## What and why

<!-- 2-4 sentences. What behaviour changes, and why, in terms of the issue's outcome. -->

## Verification

<!-- Paste from verification.md: commands run and results. Never report a check you didn't run. -->

- [ ] Backend: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- [ ] Frontend: `npm run lint && npm run build && npm test`
- [ ] `openspec validate --all` and `python scripts/check_links.py`
- [ ] **Answer quality could change?** `insight-rag-eval` run, results vs baseline: <!-- paste or "n/a" -->
- [ ] UI changed? Seen in a browser in light **and** dark mode

## Checklist

- [ ] Every acceptance criterion in the issue is covered by a test, or noted here as not covered
- [ ] `REVIEW.md` sections for the touched areas checked
- [ ] Overview / guides updated in this PR if behaviour, a contract, a boundary or a command changed (AGENTS.md rule 7)
- [ ] No secrets, real personal data, internal URLs or org names, including commit messages
