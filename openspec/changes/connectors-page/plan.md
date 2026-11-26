## Task
5.2 Runner summary writes `answerable_questions` and `unanswerable_questions` per connector; `quality.load_quality` passes them through as optional non-negative integers (absent or invalid: left out, result otherwise unchanged); tests in `test_runner.py`, `test_quality.py`, `test_quality_api.py`, including a summary without the counts

## Scope
- `backend/src/plumbline/evals/runner.py`, the `connectors={...}` block of the summary written by `run()`: add, next to `"questions": len(r["questions"])`,
  - `"answerable_questions": sum(q["answerable"] for q in r["questions"])`
  - `"unanswerable_questions": sum(not q["answerable"] for q in r["questions"])`
  `r["questions"]` entries are `asdict(QuestionScore)` plus latency, and `QuestionScore.answerable` is a bool (read in `evals/scoring.py`). `questions` (the total) stays. The full report is unchanged. The two counts add up to `questions`.
- `backend/src/plumbline/quality.py`, `_connector()`: after the existing allowlisted fields, include `answerable_questions` and `unanswerable_questions` only when each is a non-negative `int` (not a bool; no floats, no negatives, no strings). Absent or invalid: the key is left out and nothing else changes. An invalid count never makes the summary `not_measured` (design.md "Question counts, optional end to end"). If only one of the two is valid it is still left out as a pair: both or neither, because the card sentence needs both.
- New fixture `backend/tests/fixtures/quality/with_counts.json`: a copy of `both_pass.json` with synthetic counts (pdf 30 and 10, docx 28 and 12). `both_pass.json` stays as it is and serves as the "summary written before the counts existed".
- Docs, per AGENTS workflow rule 7: one sentence in `evals/README.md` (the summary paragraph) naming the two fields; the backend guide's rule 2.5 and the overview are task 7.1.
- Guides read: `docs/guides/backend-development-guide.md` (2.5 `/quality` is an allowlist from named fields; 3.2 broad `except` only at the trust boundary, which `load_quality` already is; 8 tests), `evals/README.md` for the summary description. Not an `answer-quality-change`: see eval-plan.md (output-only).

## Not in scope
- Frontend types and the card footer (6.1, 6.2).
- Any change to scoring, gating, retrieval, prompts, `/chat`, or the full report file.
- Backfilling old summary files or any default count: absent stays absent.
- A real eval run (needs models; not required by eval-plan.md).

## Files
| File | Change |
| --- | --- |
| `backend/src/plumbline/evals/runner.py` | two count fields in the summary's per-connector entry |
| `backend/src/plumbline/quality.py` | pass the pair through when both are valid |
| `backend/tests/fixtures/quality/with_counts.json` | new synthetic fixture |
| `backend/tests/test_runner.py` | summary has the counts and they add up to `questions` |
| `backend/tests/test_quality.py` | counts passed through; absent and invalid cases |
| `backend/tests/test_quality_api.py` | counts appear in `/quality`; old summary has none; no new leak |
| `evals/README.md` | one sentence |

## Tests
- Runner (spec "Counts in the summary"): in `test_summary_is_small_and_shareable`, assert `pdf["answerable_questions"] + pdf["unanswerable_questions"] == pdf["questions"] == 40`, and that `unanswerable_questions` equals the number of `answerable: false` items in the golden file (`len(UNANSWERABLE)`). This test is marked `db` and needs Postgres; it ran in this session's suite earlier.
- `load_quality` (spec "Counts passed through"): `with_counts` gives `pdf` and `docx` entries carrying both numbers.
- `load_quality` (spec "Summary without counts" and "Counts absent or invalid"): `both_pass` entries have neither key, result unchanged; parametrized in-test variants of `with_counts` written to `tmp_path` with a negative number, a float, a string, a bool, one count missing: each leaves both keys out, still `measured`, other fields identical.
- `/quality` API (spec "Counts passed through", "Summary fields only"): with the `with_counts` fixture the response entry has exactly `result`, `failed`, `metrics`, `answerable_questions`, `unanswerable_questions`; the existing no-leak test still passes (no `host`, `git_commit`, `report`, `latency_ms`, `questions`, `golden_version`).

## Checks to run
- `cd backend && uv run ruff check . && uv run ruff format --check .`
- `cd backend && uv run python -m pytest tests/test_quality.py tests/test_quality_api.py tests/test_runner.py -q`, then `uv run python -m pytest -q` (the db tests need the local Postgres; if they skip, I'll say so)
- `python scripts/check_links.py`

## Risks / Questions
- `test_runner.py` is `db`-marked; if Postgres is unreachable the runner assertion is skipped, not run. I'll report which.
- Edge: a summary where `questions` (total) disagrees with the sum of the two counts is not validated; the runner writes them together, and a hand-edited file is not a case this endpoint polices.
