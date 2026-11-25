## Context

`GET /documents` already returns `document_id`, `title`, `connector` and
`created_at`, newest first. `chunks.document_id` references
`documents(id) ON DELETE CASCADE` (migration `0001_initial.py`), so
deleting a document row removes its chunks in the same statement.
`documents.content_hash` is `UNIQUE` and ingestion dedupes by it, so once a
row is gone the same bytes index again (201). Retrieval reads only
`chunks`, so a deleted document can't be retrieved or cited. The UI is an
upload bar above the thread; frontend tests use `renderToStaticMarkup` in
Node with no jsdom.

## Decisions

### Delete is one SQL statement, no migration

- **Choice**: `DELETE FROM documents WHERE public_id = %s` in the request
  transaction; `rowcount == 0` -> 404. Cascade removes chunks.
- **Alternatives**: delete chunks explicitly first; soft delete with a
  `deleted_at` column.
- **Why**: the cascade already exists and is atomic; soft delete would
  need every retrieval query filtered and a migration, for an undo that is
  a non-goal.
- **ADR**: none (local to this change)

### Route stays thin

- **Choice**: `delete_document(conn, document_id) -> bool` next to
  ingestion's DB code; the route maps `False` to 404 with a fixed message
  ("Document not found"). Unknown and already-deleted ids are the same
  case.
- **Alternatives**: SQL inline in the route.
- **Why**: backend guide §1, routes only translate.
- **ADR**: none

### List in a dialog opened from the upload bar

- **Choice**: a "Documents (n)" button in the upload bar opens a Base UI
  `Dialog` with the list. Each row: title, type (`connector`, shown
  upper-case: PDF, DOCX), date added (locale date), and a delete button
  with `aria-label="Delete <title>"`. Delete opens a Base UI
  `AlertDialog` ("Delete <title>? It will no longer be used in answers.")
  with Cancel and Delete. Empty list: "No documents yet. Upload a PDF or
  DOCX to get started." pointing at the upload control.
- **Alternatives**: collapsible panel (pushes the thread down), sidebar
  (needs responsive layout), `window.confirm` (unstyled, ignores theme).
- **Why**: product owner choice; chat keeps full height; both components
  are installed and trap focus / restore it on close.
- **ADR**: none (follows ADR 0003's styled kit)

### List freshness

- **Choice**: fetch the list when the dialog opens and after a successful
  upload (the count on the button). No polling.
- **Alternatives**: global store; polling.
- **Why**: one user, one tab; two fetch points cover every change this UI
  makes.
- **ADR**: none

## Interfaces

```python
# plumbline/documents.py (new; or next to ingestion if it stays small)
def delete_document(conn: psycopg.Connection, document_id: str) -> bool:
    """Delete by public id; chunks go via ON DELETE CASCADE. False if absent."""

# api/app.py
@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: str, conn: Conn) -> Response:
    if not delete_document(conn, document_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)
```

```ts
// frontend/src/documents/api.ts
export type DocumentSummary = { document_id: string; title: string; connector: string; created_at: string }
export async function listDocuments(): Promise<DocumentSummary[]>
export async function deleteDocument(id: string): Promise<'deleted' | 'not-found'>  // throws on other statuses
```

## Data

No migration. Relies on `chunks.document_id ... ON DELETE CASCADE`.

## Risks / Open Questions

- An answer already streaming may cite a document deleted mid-stream;
  accepted (non-goal).
- Static-markup tests can't open dialogs or move focus: keyboard flow,
  focus ring and confirm are checked by hand in a browser (light and
  dark), recorded in `verification.md`. Colour contrast is covered by
  `design-tokens.test.ts` if only existing tokens are used.
- Base UI `Dialog`/`AlertDialog` focus trapping and return - UNVERIFIED
  for the installed version; confirm in the browser check.
