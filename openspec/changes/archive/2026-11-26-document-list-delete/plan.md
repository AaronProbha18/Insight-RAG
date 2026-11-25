## Task
3.1 Backend guide §2.1 table gains 204 and 404 rows; frontend guide notes the documents dialog and its manual checks; architecture overview lists the new endpoint. Test: `python scripts/check_links.py` passes.

## Scope
- Docs only. Guides read already for 1.1-2.3 (backend and frontend guides) and `docs/architecture/overview.md` read now. Every rule in these docs "describes how the code already works", so the wording follows what was built and verified, nothing aspirational.
- `docs/guides/backend-development-guide.md` §2.1 table: add
  - `204` | a document and its chunks were deleted (`DELETE /documents/{document_id}`)
  - `404` | `DELETE /documents/{document_id}` for an unknown or already-deleted id; fixed message "Document not found"
- `docs/guides/frontend-development-guide.md`:
  - §1.1 folder tree: add `documents/` with `api.ts` (list + delete), `DocumentList.tsx`, `Documents.tsx` (button, dialog, confirm).
  - §5 Accessibility: one bullet for the documents dialog (names: "Documents (n)" button text, `Delete <title>` aria-labels; errors in a `role="status"` live region; confirm before delete), *checked by* `DocumentList.test.tsx` / `Documents.test.tsx`.
  - §7.1: add the dialog to the "checked by hand in a browser" list: open/close, Tab order, focus ring and return, confirm/cancel, both themes. Static markup can't open dialogs.
- `docs/architecture/overview.md`:
  - diagram edge label: `POST /documents, GET/DELETE /documents, POST /chat SSE` (kept short).
  - new `### Delete: DELETE /documents/{document_id}` flow beside Ingest/Ask: `api/app.py` `remove_document` -> `ingestion.delete_document` (one `DELETE` by `public_id`; chunks go via `ON DELETE CASCADE`), **204**, or **404** when absent; retrieval reads only `chunks`, so a deleted document can't be retrieved or cited; re-uploading the same bytes indexes again (**201**).
  - "a UI component" row already allows a new folder per feature; no change.

## Not in scope
- README, AGENTS.md, ADRs, and the OpenSpec change docs (the spec is archived later by `/opsx:archive`, not here). No code.
- Documenting `GET /documents` in the status-code table (it was already there and has no status other than 200).

## Files
| File | Change |
| --- | --- |
| `docs/guides/backend-development-guide.md` | 2 table rows in §2.1 |
| `docs/guides/frontend-development-guide.md` | tree entry in §1.1; one bullet in §5; one sentence in §7.1 |
| `docs/architecture/overview.md` | diagram label; new Delete flow section |

## Tests
- `python scripts/check_links.py` passes (no new links are planned; any added anchor/link is checked by it).
- Read-through against the code for accuracy: names (`remove_document`, `delete_document`), status codes and messages match `app.py`/`ingestion.py`; component and file names match `frontend/src/documents/`.

## Checks to run
```
python scripts/check_links.py
git diff --stat      # only the three docs files
```

## Risks / Questions
- The frontend §5 bullet claims the behaviours your manual check confirmed (names, live region, confirm). I'm relying on your report that it passed; I haven't seen the browser myself.
- None otherwise.
