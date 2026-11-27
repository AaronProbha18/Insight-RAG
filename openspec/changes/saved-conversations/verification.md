## Checks Run

All run in this session on branch `feat/36-saved-conversations` at `e755e78`, with a clean working tree apart from the untracked `plan.md`. Postgres (pgvector) was running through `docker compose up -d db`.

| Command | Result |
| --- | --- |
| `cd backend; uv run ruff check .` | All checks passed |
| `cd backend; uv run ruff format --check .` | 66 files already formatted |
| `cd backend; python -m pytest -q -rs` (`uv run pytest` fails on this machine with "uv trampoline failed to canonicalize script path", so the module form was used) | 256 passed, 1 skipped (`test_models.py`: needs Ollama on localhost:11434) |
| `cd frontend; npm run lint` (PowerShell) | oxlint: no warnings |
| `cd frontend; npm run build` (`tsc -b` and vite) | built, no type errors |
| `cd frontend; npm test` (PowerShell; vitest fails under Git Bash here, also on a clean tree) | 35 files, 278 tests passed |
| `cd datamodel; tools/validate.py --model models/plumbline` (`make` is not installed; the same command) | 1 domains, 5 tables, 0 errors |
| `cd datamodel; tools/ddl.py --model models/plumbline` | wrote `build/schema.sql` (5 tables) |
| `cd datamodel; python -m pytest -q -rs tests` | 46 passed, 4 skipped (`test_drift.py` x2 and `test_postgres.py` need `DATAMODEL_TEST_DATABASE_URL` and `psql`) |
| Drift check (`db-change`): `alembic upgrade head` into one database, `build/schema.sql` into another, then `tools/drift.py` on both. `psql` is not installed here, so `drift.py`'s `psql` calls were routed through the Postgres container by a throwaway wrapper in the scratchpad; the repo is unchanged. | `no drift`, exit 0 |
| `mkdocs build --strict` (pinned `docs-requirements.txt` in a scratchpad virtualenv, output outside the repo) | built in 0.4 s, exit 0, no warnings |
| `openspec validate saved-conversations` | Change 'saved-conversations' is valid |

**Not run, so not claimed as passed:**
- The `plumbline-drift` and `datamodel` CI jobs themselves (they are not required checks; look for them green on the PR). The local drift run above used the same two inputs and `drift.py`, but not the CI workflow.
- The `llm` test in `test_models.py` and any live chat: there is no Ollama on this machine.
- Anything in a real browser (see Manual checks).

## Scenario Coverage

Backend rows run against real tables or the FastAPI test client. Frontend rows are static renders and unit tests (no jsdom).

| Spec scenario | Test |
| --- | --- |
| Conversations are stored: First question creates a conversation | `test_chat_api.py::test_done_carries_answer_id_for_answers_and_abstentions` (`done` carries `answer_id` and `conversation_id`); `test_conversations.py::test_recorder_creates_then_joins` (conversation created and titled from the question) |
| Follow-up joins the conversation | `test_chat_api.py::test_follow_up_joins_the_conversation`; `test_conversations.py::test_recorder_creates_then_joins` (second answer in the same conversation, `updated_at` moves) |
| Unknown conversation | `test_chat_api.py::test_unknown_conversation_is_404_before_answering` (404, `llm.calls == 0`, no recorder call) |
| Failed answer stores nothing | `test_chat_api.py::test_failed_answer_creates_no_conversation`, `test_error_path_records_nothing_and_has_no_answer_id`, `test_recording_failure_sends_error_not_the_answer`; `test_conversations.py::test_recorder_failure_leaves_no_conversation` |
| Conversations survive a reload: Reload restores the chat | Backend: `test_conversations_api.py::test_get_returns_the_turns`, `test_list_is_empty_then_newest_first`. Frontend: `chat/restore.test.tsx` (`turnsToMessages`, a restored thread shows answers, sources and feedback controls), `shell/chatTarget.test.ts` (`syncTarget` on reload). **Gap:** no automated test reloads a real page; see Manual checks |
| Own URL | `test_ui_mount.py::test_app_path_serves_index` (`/app/chats/conv_X` serves the index); `shell/route.test.ts` (`chatIdFor`); `shell/AppShell.test.tsx` "opening a saved chat" |
| Header shows title and scope | `chat/ChatHeader.test.tsx`; `shell/AppShell.test.tsx` "chat header" (title from the list, count, "New chat", no header on loading, not-found, error) |
| New chat keeps the previous conversation: New chat | `shell/chatTarget.test.ts` (`newChat`), `shell/AppShell.test.tsx` ("shows an empty chat for New chat"). The previous conversation staying in the list is server state; `test_conversations_api.py::test_list_is_empty_then_newest_first` covers the list, and the refresh in the UI is a manual check |
| Rename and delete: Rename | `test_conversations_api.py::test_rename_updates_the_title`; `test_conversations.py::test_rename_trims_and_updates`; `conversations/ChatList.test.tsx` (rename mode, submit); `conversations/useConversations.test.tsx` (`renameOutcome`) |
| Invalid title | `test_conversations_api.py::test_rename_rejects_invalid_titles` (422, title unchanged), `test_rename_without_a_title_is_422`; `test_conversations.py::test_rename_rejects_invalid_titles`; `test_migrations.py::test_conversation_title_length` |
| Delete | `test_conversations_api.py::test_delete_then_gone`; `conversations/ChatList.test.tsx` (confirmation and handlers); `shell/AppShell.test.tsx` ("Conversation not found" for an unknown id) |
| Feedback survives delete | `test_conversations_api.py::test_delete_keeps_the_rated_answer`; `test_conversations.py::test_delete_keeps_answers_and_feedback`; `test_migrations.py::test_deleting_a_conversation_keeps_its_answers` |
| Deleted sources are marked: Cited document deleted | `test_conversations_api.py::test_deleted_source_is_marked_and_the_answer_stays`; `test_conversations.py::test_get_marks_deleted_sources_and_keeps_the_answer`; `chat/Citations.test.tsx` ("no longer indexed"); `chat/restore.test.tsx` (note in a restored thread) |
| Chats section in the sidebar: No conversations | `shell/Sidebar.test.tsx` "Chats section" (hidden when empty or collapsed); `conversations/ChatList.test.tsx` (renders nothing when empty) |
| Accessible list | `shell/a11y.test.tsx` "keyboard reach in the Chats list" (names, links, buttons, one `aria-current`); `conversations/ChatList.test.tsx` (`aria-current`, names); `shell/a11y.test.tsx` source scans include `ChatList.tsx` and `ChatHeader.tsx`; `design-tokens.test.ts` (header text tokens measured at 4.5:1, light and dark) |
| app-shell: Return to an unchanged chat | `shell/AppShell.test.tsx` ("keeps the chat mounted but hidden on the Documents page"); `shell/chatTarget.test.ts` (target unchanged while the URL names the same chat). **Gap:** a live visit to Documents and back is a manual check |
| app-shell: New chat clears it | as "New chat" above |

Every scenario has at least one automated test. The rows marked Gap or manual rely on unit-level evidence because there is no jsdom or browser test runner in this repo.

## Eval Results

N/A per eval-plan: the graph, retrieval and prompt are unchanged; answers are only grouped into conversations after they are produced. The issue's `answer-quality-change` label is therefore satisfied by that statement, and no eval was run. (An eval could not run here in any case: no model server.)

## Deviations

- **Index direction:** `design.md` originally wrote `conversations (updated_at DESC)`. The datamodel index schema has no sort direction, so `plumbline-drift` would flag it. Revision `0003` uses a plain `(updated_at)` index and `design.md` was updated to match (agreed at task 1.1).
- **Restored citations** carry no `[cN]` id and no quoted text: `answers.citations` keeps only `document_id`, `title`, `filename` and `locator` (design D1, #12). The UI shows `title - locator` with no expander and the "no longer indexed" note (option A, agreed at task 3.2). Inline `[c3]` markers in a restored answer have no matching label.
- **Recorder** replaced the bare recording lambda behind `get_recorder`; it also answers `resolve(conversation_id)`, so the 404 check happens before the graph runs.
- **Delete of the open chat** goes to New chat (decision 2 of task 3.4); the spec is silent on it.
- **Sidebar Chats section hidden when the sidebar is collapsed** (decision 3 of task 3.4); the spec is silent on it.
- **Rename bumps `updated_at`**, so a renamed chat moves to the top of the list; the design is silent on it.
- Existing tests edited because the new UI made their markers ambiguous: three `AppShell` tests that used `<h1` as a stand-in for the Documents page now look for `>Documents</h1>` and `>Connectors and quality</h1>`.

## Verdict

PASS on everything that can be run here. Not yet verified, and needed before merge:

**Manual checks (real browser, backend and a model running):**
1. Ask a question in a new chat: the URL becomes `/app/chats/{id}` and the chat appears in the sidebar; reload restores the answer, its sources and the feedback controls.
2. New chat shows an empty thread and the previous chat stays in the list; open Documents and return: the chat is unchanged.
3. Rename a chat (Enter saves, Escape cancels): the sidebar and the header update. Delete another chat (confirm), then delete the open one: it goes to a new chat; its old URL shows "Conversation not found".
4. Delete a cited document, reopen the chat: the answer stays and its source says "Source no longer indexed".
5. Layout: the thread still fills the height under the header; resize, long thread, scroll to the bottom; light and dark themes; the sidebar drawer below `md`.

**CI to watch on the PR:** `plumbline-drift` and the `datamodel` job (not required checks).
