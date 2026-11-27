Issue: #37

## Why

When Plumbline refuses a question or a person marks an answer wrong, the owner has no view of it, so they cannot tell which document is missing and the same question is refused again. Both signals are already stored (`answers.abstained`, `feedback.rating = 'down'`) but nothing reads them back.

## What Changes

- New read-only `GET /unanswered?days=` endpoint: refused answers and thumbs-down answers in the range, grouped by question after ignoring case and extra spaces, with count, last date and reason.
- New **Unanswered** page at `/app/unanswered`: one row per group, a date range filter (Last 7 / 30 / 90 days, All time; default Last 30 days), an empty state, and an **Add a document** link to the Documents page.
- Sidebar item **Unanswered** showing the open count (groups in the last 30 days), refreshed after each chat answer.
- After a refusal, the chat message links to the page ("Added to Unanswered").
- Docs: architecture overview flow and route table.

## Non-Goals

- Suggesting which document to add.
- Exporting the list.
- Changing how refusals or feedback are decided.
- Closing a group, by dismissal or automatically (triage decision; a follow-up issue can add it).
- A new table or migration: the existing `answers` and `feedback` rows are the record.

## Capabilities

### New

- `unanswered-questions`: list refused and wrong-marked questions, grouped, on a page reached from the sidebar and from refusals in the chat.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Reads stored answers only; the `/chat` path, refusal logic and feedback are untouched. |
| Connector / ingestion | No | Not touched. |
| Auth / document ACLs | No | Single-user local app; lists only questions already stored locally. |
| Public API or stream contract | Yes | New `GET /unanswered` endpoint; the `/chat` stream is unchanged. |
| Persistence / migration | No | Read-only queries over `answers` and `feedback`; no revision. |
| New external dependency or service | No | None. |

## Impact

`backend/src/plumbline/unanswered.py` (new), `backend/src/plumbline/api/app.py`, `frontend/src/unanswered/` (new), `frontend/src/shell/` (route, sidebar), `frontend/src/App.tsx`, the chat message rendering, `docs/architecture/overview.md`.
