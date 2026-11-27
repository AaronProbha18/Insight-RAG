Issue: #36

## Why

A chat lives only in memory, so reloading the page or pressing New chat loses it, and a person cannot return to an earlier answer or its sources.

## What Changes

- New `conversations` table; `answers` gains a nullable `conversation_id` (`ON DELETE SET NULL`). Alembic revision 0003 and the `datamodel/models/plumbline` mirror.
- `POST /chat` accepts an optional `conversation_id`. Without one, the first question creates a conversation titled from that question; the `done` event also carries `conversation_id`.
- New endpoints: `GET /conversations`, `GET /conversations/{id}` (turns, each citation with `indexed`), `PATCH /conversations/{id}` (rename), `DELETE /conversations/{id}`.
- Sidebar **Chats** section (hidden when empty), newest first, with rename and delete per row.
- Each conversation has its own URL, `/app/chats/{id}`; opening it restores questions, answers and citations. An unknown or deleted id shows "Conversation not found".
- Page header shows the conversation title and a read-only scope label, "Searching all N documents".
- New chat opens an empty conversation; the previous one stays listed.
- A citation whose document was deleted says the source is no longer indexed.
- Docs: architecture overview flows and the frontend/backend guides.

## Non-Goals

- Syncing conversations between machines or users (no accounts exist).
- Search across conversations.
- Changing how answers are produced: each question is still answered on its own, with no earlier turns sent to the model.
- Narrowing a chat to selected documents: the scope label is read-only (issue OPEN QUESTION, decided in triage).
- A cap on kept conversations or auto-delete (issue OPEN QUESTION, decided in triage: no limit).

## Capabilities

### New

- `saved-conversations`: store, list, open, rename and delete conversations, through the API and the sidebar.

### Modified

- `app-shell`: "Chat survives navigation" changes, because New chat now keeps the previous conversation instead of clearing it.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | The graph, retrieval and prompt are unchanged; only the question and result are grouped after answering. |
| Connector / ingestion | No | Not touched. |
| Auth / document ACLs | No | Single-user app with no accounts; conversations are as visible as documents already are. |
| Public API or stream contract | Yes | `/chat` request gains optional `conversation_id`; `done` gains `conversation_id`; four new `/conversations` routes. All additive. |
| Persistence / migration | Yes | New `conversations` table and a nullable FK on `answers` (revision 0003). |
| New external dependency or service | No | Postgres, FastAPI and React already in use. |

## Impact

- Backend: `backend/migrations/versions/0003_conversations.py`, new `backend/src/plumbline/conversations.py`, `api/app.py`, `feedback.py` (`record_answer` takes a conversation), tests.
- Datamodel: `datamodel/models/plumbline/domains/rag/tables/` (new `conversations.yaml`, `answers.yaml`), regenerated `build/`.
- Frontend: `frontend/src/chat/` (adapter, ChatPane), new `frontend/src/conversations/`, `shell/` (Sidebar, route, AppShell), tests.
- Docs: `docs/architecture/overview.md`, frontend and backend guides.
