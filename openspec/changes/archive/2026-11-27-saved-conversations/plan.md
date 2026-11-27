## Task
4.1 `docs/architecture/overview.md` (diagram routes, Ask flow, new Conversations flow), backend and frontend guides; `mkdocs build --strict` passes.

## Scope
Docs only, written from the code as committed (tasks 1.1 to 3.5), not from the design. Every claim below is one I checked in the code or tests during those tasks; anything I cannot point to I leave out. Read: `overview.md`, both guides, `docs-site.md`, `mkdocs.yml` and `docs-requirements.txt`.

**`docs/architecture/overview.md`**
- Mermaid: the browser edge also lists `GET, PATCH, DELETE /conversations`; a `conversations.py` node between the API and the database.
- "Where things live": a row for saved conversations (`backend/src/plumbline/conversations.py`); the UI row names the `conversations/` folder.
- **Ask flow**: the request may carry `conversation_id`; an unknown one is **404** before retrieval or the model runs; the answer is recorded through the `Recorder` in one transaction, creating the conversation (titled from the question) on the first answer; `done` carries `answer_id` and `conversation_id`; on failure nothing is created.
- **New flow, "Saved conversations: `/conversations`"**: list (newest updated first), get (turns in order, each citation with `indexed`, true while its document still exists), rename (trimmed, 1 to 200 characters, else **422**; **404** unknown), delete (**204**; answers and feedback stay, their `conversation_id` becomes null). Each call is one function in `conversations.py`.
- **Open a page**: `/app/chats/{id}` is the chat page for a saved conversation; after the first answer of a new chat the URL is replaced with it (no history entry) and the sidebar list refreshes; New chat goes to `/` with an empty thread; the chat is kept while the person visits Documents or Connectors.
- **Data model**: a `conversations` row (`conv_<ULID>`, title 1 to 200, `updated_at`); `answers.conversation_id` with `ON DELETE SET NULL`; the "Stored locally" paragraph also covers titles (the first question) and points at OQ-ANSWERS-001.

**`docs/guides/backend-development-guide.md`**
- 2.1 status table: rows for `GET /conversations` and `GET /conversations/{id}` (200, 404 "Conversation not found"), `PATCH` (200, 404, 422), `DELETE` (204, 404), and `/chat` 404 for an unknown `conversation_id`.
- 2.2: `conversation_id` (`conv_<ULID>`) is a public id; the internal key never leaves.
- 2.4: the `done` payload and the optional request field; the order "unknown id -> 404, then answer, then record".
- 8.1a: the fake recorder also has `resolve(conversation_id)` and returns `(answer_id, conversation_id)`.

**`docs/guides/frontend-development-guide.md`**
- 1.1 tree: `conversations/` (`api.ts`, `useConversations.ts`, `useConversation.ts`, `ChatList.tsx`, `saveAction.ts`), `chat/restore.ts` and `ChatHeader.tsx`, `shell/chatTarget.ts`; the `App.tsx` and `Sidebar.tsx` lines (chat target, Chats section).
- 1.4: `/app/chats/{id}` and `chatIdFor`, `replace`.
- 2.1: `createChatAdapter(conversation)` replaces `chatAdapter`; `ChatPane` props; the target state (`syncTarget`, `newChat`, `adopt`) replaces "New chat gives `ChatPane` a new `key`"; `onConversation` is captured on mount.
- 2.2: a restored citation has no `id` or `text`, so `Citations` shows `title - locator` with no expander, plus "Source no longer indexed" when `indexed` is false.
- 7: static tests cannot show a restored thread's metadata or the list refreshing; say what is checked by hand.

## Not in scope
- AGENTS.md layout, README, the Docker docs, ADRs (none are changed by this feature).
- Syncing specs and archiving the change (separate step before merge, per the workflow).

## Files
| File | Change |
| --- | --- |
| `docs/architecture/overview.md` | as above |
| `docs/guides/backend-development-guide.md` | as above |
| `docs/guides/frontend-development-guide.md` | as above |

## Tests
- `mkdocs build --strict` passes (a broken link or anchor fails it).
- A search for stale text: `chatAdapter`, `chatKey` and "a new `key`" in `docs/` return nothing outdated; every file path I add to the docs exists (`git ls-files` check).

## Checks to run
- Docs tooling is not installed here. I will create a virtualenv in the scratchpad (outside the repo), `pip install -r docs-requirements.txt` (pinned `mkdocs==1.6.1`, `mkdocs-material==9.8.2`, from PyPI), and run `mkdocs build --strict` from the repo root with the output directory in the scratchpad, so nothing lands in the repo.
- `git diff --stat` shows only the three docs files.

## Risks / Questions
- If the install cannot reach PyPI, I cannot run the strict build and will say so; CI runs it (`ci.yml`).
- The doc text describes behaviour I could not exercise end to end (no model here). I will only state what the code and tests show, and keep the "hand check" notes in the frontend guide honest.
