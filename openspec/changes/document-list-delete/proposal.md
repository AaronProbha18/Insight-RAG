Issue: #12

## Why

People can upload documents but can't see what is indexed or remove
anything, so a wrong or outdated upload stays in every answer until the
database volume is wiped.

## What Changes

- New `DELETE /documents/{document_id}`: 204 on success, 404 (no internal
  details) when the id is unknown or already deleted. The document row and
  all its chunks go in one transaction.
- Chat UI: a "Documents (n)" button in the upload bar opens a dialog
  listing indexed documents (title, type, date added, newest first), with
  an empty state pointing to upload.
- Each row has a delete control that asks for confirmation in an alert
  dialog; on confirm the row disappears.
- The list refreshes when the dialog opens and after a successful upload.
- Docs: backend API conventions table gains 204/404; frontend guide notes
  the dialogs.

## Non-Goals

- Bulk delete, undo or a trash.
- Editing or re-indexing a document.
- Per-user permissions.
- Pagination.
- Stopping an answer that is already streaming when its source is deleted.

## Capabilities

### New

- `document-management`: listing and deleting indexed documents, via API
  and chat UI.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | Yes | Retrieval logic unchanged, but deleting removes sources from answers; a deleted document must never be cited again. |
| Connector / ingestion | No | Ingestion unchanged; re-upload after delete works through the existing hash dedupe. |
| Auth / document ACLs | No | No per-user permissions (non-goal); anyone with the UI can delete, as anyone can upload today. |
| Public API or stream contract | Yes | New `DELETE /documents/{document_id}` endpoint. `/chat` unchanged. |
| Persistence / migration | Yes | Deletes rows; relies on existing `ON DELETE CASCADE` on `chunks.document_id`. No migration. |
| New external dependency or service | No | Uses `@base-ui/react` Dialog and AlertDialog, already installed. |

## Impact

`backend/src/plumbline/api/app.py`, backend tests,
`frontend/src/` (new `documents/` folder, upload bar), frontend tests,
`docs/guides/backend-development-guide.md`,
`docs/guides/frontend-development-guide.md`, `docs/architecture/overview.md`.
