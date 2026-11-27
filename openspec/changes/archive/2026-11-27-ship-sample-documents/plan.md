## Task
3.1 Document the samples path, `PLUMBLINE_SAMPLES_DIR` and `POST /documents/samples` (request, response, 404) in `docs/guides/backend-development-guide.md`, and add a line in `docs/architecture/overview.md` if that page lists endpoints.

## Scope
- Backend guide: add a section 2.7 "`/documents/samples` indexes the shipped samples" (no body; `PLUMBLINE_SAMPLES_DIR`, `/app/samples` in the image, default `evals/connectors/pdf/corpus` in dev; response `{indexed, failed}`; 200 even with failures; 404 naming the path; idempotent via content hash; only `ParseError`/`UnsupportedFileType` land in `failed`; *Enforced by* the four `test_samples_*` tests). Add a 404 row to the 2.1 status table.
- Overview: it lists endpoints (line 14 diagram label and per-endpoint Flows), so add `POST /documents/samples` to the diagram label and a short "Index the samples" flow under Ingest.
- Guide read: backend guide.

## Not in scope
- README, frontend guide, any code.

## Files
| File | Change |
| --- | --- |
| `docs/guides/backend-development-guide.md` | section 2.7 + status-table row |
| `docs/architecture/overview.md` | diagram label + short flow |

## Tests
- `backend/tests/test_check_links.py` and the docs link check keep passing; `mkdocs build` if available.

## Checks to run
- `cd backend && uv run python -m pytest tests/test_check_links.py -q`
- `uv run ruff check . && uv run ruff format --check .` (no code touched; skip if unchanged)

## Risks / Questions
- none
