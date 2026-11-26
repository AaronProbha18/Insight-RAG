## 1. Backend

- [x] 1.1 `delete_document(conn, document_id) -> bool` and `DELETE /documents/{document_id}` (204 / 404 "Document not found"). Test: `test_documents_api.py` - delete returns 204 and the document and its chunks are gone; unknown id and second delete return 404 with the fixed message; re-upload of the same bytes after delete returns 201.
- [x] 1.2 Deleted documents are never used: integration test - ingest two documents, ask a question only one answers (answers with a citation), delete it, ask again -> abstains, no citation, retrieval returns no chunk from it. Test: the eval-plan command passes.

## 2. Frontend

- [x] 2.1 `frontend/src/documents/api.ts`: `listDocuments()` and `deleteDocument(id)`. Test: `api.test.ts` with stubbed `fetch` - list parse, 204 -> deleted, 404 -> not-found, 500 and network error throw.
- [x] 2.2 `DocumentList` component: rows (title, type, date), newest first as returned, delete button with `aria-label`, empty state pointing to upload. Test: `DocumentList.test.tsx` static markup - two rows in order with all fields; empty state text; accessible names present; title is HTML-escaped.
- [x] 2.3 "Documents (n)" button in the upload bar opening a `Dialog` with `DocumentList`; delete via `AlertDialog` confirm; refetch on open and after a successful upload. Test: static markup of the button with its count and accessible name; `npm run lint && npm run build && npm test`; keyboard/focus/confirm and both themes checked by hand in a browser.

## 3. Docs

- [x] 3.1 Backend guide §2.1 table gains 204 and 404 rows; frontend guide notes the documents dialog and its manual checks; architecture overview lists the new endpoint. Test: `python scripts/check_links.py` passes.
