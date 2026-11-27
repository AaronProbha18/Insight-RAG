## 1. Storage

- [x] 1.1 Alembic revision `0003_conversations` (table, index, `answers.conversation_id` `ON DELETE SET NULL`, downgrade) and the `datamodel/models/plumbline` mirror (new `conversations.yaml` referencing OQ-ANSWERS-001, `answers.yaml` column, regenerated `build/`); test: upgrade/downgrade round-trip, and `plumbline-drift` passes.
- [x] 1.2 `conversations.py`: `create` (title rule D5), `resolve`, `touch`, `list_all`, `get` (with `indexed` per citation), `rename` (1-200 after trim), `delete`; tests for each, including title truncation, deleted-source `indexed: false`, and delete leaving answers and feedback.

## 2. API

- [x] 2.1 `POST /chat` takes optional `conversation_id`: unknown -> 404 before answering; none -> create on the first recorded answer in the same transaction; `record_answer` stores the conversation and touches `updated_at`; `done` carries `conversation_id`; failure creates nothing. Tests for all four "Conversations are stored" scenarios.
- [x] 2.2 `GET /conversations`, `GET /conversations/{id}`, `PATCH /conversations/{id}`, `DELETE /conversations/{id}` with 200/204/404/422 as in design; tests for each status, newest-first order and the deleted-source scenario.

## 3. Frontend

- [x] 3.1 `conversations/api.ts` and `useConversations` hook (list, get, rename, delete, refresh); tests against mocked fetch.
- [ ] 3.2 Chat adapter sends `conversation_id` and reads it from `done`; `ChatPane` takes initial turns (citations and `answer_id` in metadata) so restored answers show citations and feedback. Test confirms the D7 UNVERIFIED `initialMessages` shape, and that a "no longer indexed" note shows for `indexed: false`.
- [ ] 3.3 Routing: `/app/chats/{id}` page; after the first `done` in a new chat the URL becomes `/app/chats/{id}` and the list refreshes; New chat goes to `/` with an empty thread and keeps the list; unknown id shows "Conversation not found". Tests for reload-restore, New chat and not-found.
- [ ] 3.4 Sidebar Chats section (hidden when empty, newest first, `aria-current` on the open one) with rename and delete per row; tests for hidden-when-empty, rename and delete updating the list, and keyboard reachability in `shell/a11y.test.tsx`.
- [ ] 3.5 Chat page header: conversation title and read-only "Searching all N documents"; test for both, and contrast for the new text tokens in light and dark (`design-tokens.test.ts`).

## 4. Docs

- [ ] 4.1 `docs/architecture/overview.md` (diagram routes, Ask flow, new Conversations flow), backend and frontend guides; `mkdocs build --strict` passes.
