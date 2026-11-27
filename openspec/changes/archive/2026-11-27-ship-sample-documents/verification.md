## Checks Run

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | All checks passed |
| `cd backend && uv run ruff format --check .` | 62 files already formatted |
| `cd backend && uv run python -m pytest -q -rs` (Postgres up via `docker compose up -d db`) | 219 passed, 1 skipped (`test_models.py:66`, needs Ollama) |
| `openspec validate ship-sample-documents` | valid |
| `docker build -q -t plumbline-samples-check .` | built |
| `docker run --rm plumbline-samples-check sh -c "ls /app/samples; echo ---; ls /app"` | see below |
| `docker run --rm plumbline-samples-check sh -c "find / ( -name corrupt -o -name 'golden*' -o -name 'baseline*' ) ..."` | only `/app/backend/src/plumbline/evals/golden.py` (application source, not eval data) |
| `mkdocs build --strict` | NOT RUN: `mkdocs` not installed locally; pending CI. `tests/test_check_links.py` passed (5 passed) after the docs edit |
| frontend lint/build/test | NOT RUN: no frontend files changed |
| datamodel drift check | N/A: no `db-change`, no Alembic revision |

Image listing:

```
employee-handbook.pdf
meeting-notes-2026-q3.pdf
returns-policy.pdf
trail-ebike-manual.pdf
---
backend
samples
static
```

Not run: an end-to-end `POST /documents/samples` against the running container.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Samples present at the documented path | image listing above (`/app/samples` holds the four PDFs) |
| No eval data in the image | image listing above (`/app` has no `evals`; no golden sets, baselines or `corrupt/`) |
| All samples indexed | `tests/test_documents_api.py::test_samples_all_indexed` |
| Idempotent | `tests/test_documents_api.py::test_samples_idempotent` |
| One sample fails to parse | `tests/test_documents_api.py::test_samples_corrupt_file_in_failed` |
| Samples directory missing | `tests/test_documents_api.py::test_samples_dir_missing_404` |

## Eval Results

N/A per eval-plan (ingestion path unchanged; retrieval, prompts and graph untouched). Not labelled `answer-quality-change`, so no eval was run.

## Deviations

none

## Verdict

PASS (with `mkdocs build --strict` pending CI)
