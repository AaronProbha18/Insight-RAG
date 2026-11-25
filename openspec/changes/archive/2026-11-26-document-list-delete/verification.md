## Checks Run

Run in this session on branch `feat/12-document-list-delete`. Backend DB tests ran against the local Postgres container with `DATABASE_URL` set to its pre-rename role (the container predates the rename and rejects the new default credentials); each run uses a throwaway database, so dev data is untouched.

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check . && uv run ruff format --check .` | pass: "All checks passed!", 53 files formatted |
| `cd backend && uv run pytest -q -rs` | pass: 147 passed, 1 skipped (`test_models.py:66`, needs Ollama on :11434; unrelated to this change) |
| `cd D:\...\frontend && npm run lint` | pass: exit 0, no warnings |
| `cd D:\...\frontend && npm run build` | pass: exit 0 (`tsc -b` + Vite build) |
| `cd D:\...\frontend && npm test` | pass: 13 files, 44 tests |
| `python scripts/check_links.py` | pass: links ok (73 files) |
| `openspec validate document-list-delete` | pass: change is valid |
| `openspec validate --all` | pass: 4 passed, 0 failed |

Manual browser check (light and dark): **performed and reported as passed by the user**, not by Claude, who cannot drive a browser. Run against the real API on a throwaway database (`plumbline_manual`, 8-dim vectors, test fake embeddings, no LLM; since dropped) and the Vite dev server. Covered: list order and empty state, keyboard-only open/close, Tab order, visible focus ring, focus return, confirm and cancel, delete and count update, count after upload, load-failure message, and text contrast in both themes. Detailed notes were not recorded.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Document list: Two documents | `frontend/src/documents/DocumentList.test.tsx` "shows title, type and date for each document, in the order given"; manual check (open dialog, newest first) |
| Document list: No documents | `DocumentList.test.tsx` "points to upload when empty"; manual check |
| Delete a document: Delete an indexed document | `backend/tests/test_documents_api.py::test_delete_document_204_removes_document_and_chunks` (204, document and chunks gone, other document intact); `frontend/src/documents/api.test.ts` ("reports deleted on 204"); confirm-then-delete flow by manual check |
| Delete a document: Unknown or already-deleted id | `test_documents_api.py::test_delete_unknown_and_repeat_404` (404, fixed message); `api.test.ts` ("reports not-found on 404") |
| Deleted documents are never used again: Question only the deleted document answered | `test_documents_api.py::test_deleted_document_is_never_retrieved_or_cited` (cited before delete; abstains, no citations, no retrieved chunk after) |
| Re-upload after delete: Same bytes again | `test_documents_api.py::test_reupload_after_delete_201` |
| Accessible list and delete controls: Keyboard only | `DocumentList.test.tsx` ("gives every delete button an accessible name"), `Documents.test.tsx` (button name `Documents (0)`), `src/design-tokens.test.ts` (contrast of the existing tokens used); keyboard, focus ring, confirm and both themes by manual check |

No scenario is without a test or a recorded manual check. Static-markup tests cannot open dialogs or move focus, so the keyboard, focus and confirm parts of the last scenario, and the UI half of "Delete an indexed document", rest on the manual check.

## Eval Results

Eval-plan applies narrowly: deterministic test, no judge run, and the product owner decided not to re-run the full `plumbline-eval` (see `eval-plan.md`). Command run: `cd backend && uv run pytest tests/test_documents_api.py -k "deleted"` -> 1 passed.

| Metric | Baseline | Result | Pass rule met? |
| --- | --- | --- | --- |
| Chunks retrieved from a deleted document | n/a (delete is new) | 0 | yes |
| Citations to a deleted document | n/a | 0 | yes |
| Abstains on a question only the deleted document answered | n/a | yes (`abstain`, `done`; LLM not called again) | yes |

The golden-set metrics were not re-run. Retrieval, ranking, chunking, embeddings, prompts and the graph are unchanged in this diff, so those metrics cannot move for an unchanged corpus.

## Deviations

- Backend tests ran with `DATABASE_URL` pointed at the container's pre-rename role (environment issue, not a code change). CI uses its own Postgres.
- Base UI `AlertDialog` nested inside `Dialog` (UNVERIFIED in the design) was confirmed only by the user's manual check; there is no automated test for it.
- No other deviations from design or specs. Delete is a single `DELETE` with cascade, no migration, as designed; `delete_document` lives in `ingestion.py` (the design allowed this).

## Verdict

PASS
