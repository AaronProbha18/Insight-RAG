## Task
3.1 Backend guide (endpoint, allowlist), frontend guide (quality dialog, manual checks), architecture overview (`GET /quality`, summary file), `evals/README.md` (`PLUMBLINE_EVAL_SUMMARY`, sampled runs don't count, Docker mount). Test: `python scripts/check_links.py` passes.

## Scope
Docs only, describing what 1.1 to 2.3 built (every claim below was verified in code in those tasks). Each edit is a few lines in the file's existing style; no rewrites.
- `docs/guides/backend-development-guide.md`
  - 2.1 status table: one row, `200 | GET /quality always returns 200; no summary is "not_measured", never an error`.
  - 2.2/2.3 area: one short rule, "**2.5 `/quality` is an allowlist.**" `quality.load_quality` builds the response from named fields; `host`, `git_commit`, `report`, `reason`, latency and the other summary fields are never returned; only the three gated metrics; PASS/FAIL comes from the runner, not recomputed. *Enforced by* `tests/test_quality.py` (`test_no_internal_fields_leak`) and `tests/test_quality_api.py`.
  - 1.4 config owners: `api/app.py` reads `PLUMBLINE_EVAL_SUMMARY` (already a composition root; one mention only if the rule lists variables).
- `docs/guides/frontend-development-guide.md`
  - 1.1 tree: add `quality/ api.ts GET /quality`, `QualityPanel.tsx`, `Quality.tsx "Quality: <state>" button and dialog`.
  - 5 accessibility: a "**The quality dialog:**" bullet (button text is its name; PASS/FAIL words plus `aria-hidden` icons, never colour alone; mismatch warning in `role="status"`; *Checked by* `quality/QualityPanel.test.tsx`, `quality/Quality.test.tsx`).
  - 7.1 manual checks: add the quality dialog to the by-hand list (open/close, Tab order, focus return, focus ring, light and dark).
- `docs/architecture/overview.md`
  - mermaid edge label: add `GET /quality`.
  - "Adding..." table: row for the quality summary (`backend/src/plumbline/quality.py`).
  - Flows: a short "### Show the last eval: `GET /quality`" section: reads `PLUMBLINE_EVAL_SUMMARY` (default `evals/reports/latest.summary.json`) per request; `not_measured` when missing/invalid, exit 2 or sampled; otherwise allowlisted fields plus a `config_matches` check of chat model, embedding model and abstain threshold (judge not compared); no database, no model.
- `evals/README.md`
  - After the summary paragraph: `PLUMBLINE_EVAL_SUMMARY` (the quality panel reads it; default path), sampled runs (`--sample`) and exit 2 show "Not measured yet", and Docker: the image has no `evals/`, so mount the summary file into the container and set `PLUMBLINE_EVAL_SUMMARY` to the mounted path (a `docker run -v`/compose `volumes:` example in prose, not a compose file edit). UNVERIFIED: the exact container path used by the image's working directory; confirm in the `Dockerfile` before writing the example, and use a path I can verify.

## Not in scope
- Code, tests, `docker-compose.yml`, `Dockerfile` (a compose volume is documented, not added).
- `.env.example` (done in 1.2; I can't read it, see Risks).
- Other docs not named in the task (README.md, AGENTS.md layout).

## Files
| File | Change |
| --- | --- |
| `docs/guides/backend-development-guide.md` | status row, allowlist rule |
| `docs/guides/frontend-development-guide.md` | tree, accessibility bullet, manual checks |
| `docs/architecture/overview.md` | diagram label, table row, flow section |
| `evals/README.md` | env var, sampled/exit-2 note, Docker mount |

## Tests
- `python scripts/check_links.py` passes (the task's test).
- `cd backend && uv run pytest tests/test_check_links.py -q` (the checker's own tests), and the full backend `pytest` stays green (some docs are asserted by tests such as `test_old_names.py`, so a stray old name in new text would fail it).
- Read-through of each new claim against the code: status 200 always, allowlist fields, the three config comparisons, default path.

## Checks to run
- `python scripts/check_links.py`
- `cd backend && uv run ruff check . && uv run pytest -q`

## Risks / Questions
- Docker mount path: the image layout decides the example path. I will read the `Dockerfile` first and document only what it shows; if it doesn't pin a path I'll describe the setting without a concrete path rather than invent one.
- The `.env.example` entry's wording is yours and unread by me; docs will state the variable's meaning and default from the code, not from that file.
- Wording check: docs avoid "confidence" language and real data; the examples use only fixture-style values.
