Issue: #33

## Why

Plumbline is one page with an upload bar whose buttons open dialogs. New
features have nowhere to go, and nothing has its own URL to link to or come
back to.

## What Changes

- A persistent left sidebar. Top to bottom: the logo and a collapse button,
  **New chat**, **Knowledge** (Documents with the document count, and
  Connectors shown as not yet available), **Insights** (Quality, which opens
  the existing Quality dialog), and a footer link to the architecture docs.
- The main area shows one page at a time. Chat is at `/` and Documents at
  `/app/documents`. Browser back and forward and shared links work.
- The backend serves `index.html` for any `GET /app/{path}`, so a reload or
  a shared link on a page URL works in production (SPA fallback).
- The Documents page replaces the Documents dialog: upload area, a search
  by title, a filter by type (All, PDF, Word), and the list with delete.
  It reuses the existing API and components.
- The chat stays in memory while the person moves between pages.
- Below about 768 px the sidebar collapses behind a menu button. On wider
  screens the collapse button shrinks it to icons, and the choice survives
  a reload.
- The upload bar and the Documents dialog are removed. The Quality dialog
  stays, opened from the sidebar.

## Non-Goals

- Saved conversation history and a chat list in the sidebar.
- The Connectors page (separate issue). The sidebar entry shows "coming
  soon" and is not a link.
- The Unanswered and Settings pages.
- Showing the running chat model's name. No API exposes it today, and adding
  one is a separate API change (follow-up issue).
- Changing how answers are produced.

## Capabilities

### New

- `app-shell`: sidebar, page routing with URLs, SPA fallback, collapse and
  responsive behaviour, and accessibility.

### Modified

- `document-management`: the document list becomes a page with upload,
  search and a type filter.
- `chat-ui`: upload moves from the bar above the chat to the Documents page.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | UI layout only; `/chat` is untouched |
| Connector / ingestion | No | Upload calls the same `POST /documents` |
| Auth / document ACLs | No | No auth exists; the fallback serves only the public `index.html` |
| Public API or stream contract | Yes | New `GET /app/{path}` returning `index.html`; existing JSON routes unchanged |
| Persistence / migration | No | Collapse choice lives in browser `localStorage` only |
| New external dependency or service | No | Built-in router on the History API; no router library |

## Impact

- `frontend/src/`: new `shell/` (sidebar, router hook), new
  `documents/DocumentsPage.tsx`, `App.tsx` reworked; `upload/Upload.tsx`
  reused inside the page; `documents/Documents.tsx` (dialog) removed.
- `backend/src/plumbline/api/app.py`: SPA fallback route, with a test.
- `docs/guides/frontend-development-guide.md` (code organisation, routing)
  and `docs/architecture/overview.md` where it describes the UI.
