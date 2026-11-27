Issue: #35

## Why

A new user with no indexed documents lands on an empty chat whose questions can only be refused, and nothing tells them to add documents first.

## What Changes

- `useDocuments` exposes `loaded`, true once the first `GET /documents` settles, so the chat page renders nothing until the list arrives (no first-run flash).
- New `FirstRun` component in the chat area: Plumbline mark, "Ask questions about your documents", one line on source citations, **Upload documents** (navigates to `/app/documents`), **Try the sample documents**, and three steps (add documents, ask a question, check the source).
- **Try the sample documents** calls the existing `POST /documents/samples`, then refreshes the document list. If `indexed` is empty it stays on the first-run screen with an error; otherwise the chat shows with one suggested question about the samples (from `returns-policy.md`).
- `AppShell` shows `FirstRun` when the list loaded and is empty, and the chat otherwise; a failed list load shows the chat with the existing error. Deleting every document already refreshes the list, so the first-run screen returns.
- Docs: the user guide describes the first-run screen.

## Confirmed Low-Risk

Frontend and docs only: it calls the existing, idempotent `POST /documents/samples` from #40 and changes no retrieval, prompt, connector, ingestion, auth, API contract or persistence.

## Impact

- `frontend/src/documents/useDocuments.ts`, `frontend/src/documents/api.ts`
- `frontend/src/chat/FirstRun.tsx` (new), `frontend/src/chat/ChatPane.tsx`
- `frontend/src/shell/AppShell.tsx`, `frontend/src/App.tsx`
- Tests alongside each; `frontend/src/shell/a11y.test.tsx`
- `docs/` user guide page
