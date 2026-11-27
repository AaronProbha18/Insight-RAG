## Checks Run

All run in this session on branch `feat/36-saved-conversations` at `ff89878`, with a clean working tree apart from the untracked `plan.md`. Postgres (pgvector) was running through `docker compose up -d db`, and Ollama through the existing `plumbline-ollama` container.

| Command | Result |
| --- | --- |
| `cd backend; uv run ruff check .` | All checks passed |
| `cd backend; uv run ruff format --check .` | 66 files already formatted |
| `cd backend; python -m pytest -q -rs` (`uv run pytest` fails on this machine with "uv trampoline failed to canonicalize script path", so the module form was used) | 256 passed, 1 skipped (`test_models.py`: needs `PLUMBLINE_LLM` and `PLUMBLINE_EMBEDDINGS` set to Ollama models, which were not set in this run) |
| `cd backend; PLUMBLINE_LLM=ollama:llama3.1:8b PLUMBLINE_EMBEDDINGS=ollama:nomic-embed-text python -m pytest -q -rs tests/test_models.py` (the gated test, with Ollama up) | 7 passed |
| `cd frontend; npm run lint` (PowerShell) | oxlint: no warnings |
| `cd frontend; npm run build` (`tsc -b` and vite) | built, no type errors |
| `cd frontend; npm test` (PowerShell; vitest fails under Git Bash here, also on a clean tree) | 35 files, 282 tests passed |
| `cd datamodel; tools/validate.py --model models/plumbline` (`make` is not installed; the same command) | 1 domains, 5 tables, 0 errors |
| `cd datamodel; tools/ddl.py --model models/plumbline` | wrote `build/schema.sql` (5 tables) |
| `cd datamodel; python -m pytest -q -rs tests` | 46 passed, 4 skipped (`test_drift.py` x2 and `test_postgres.py` need `DATAMODEL_TEST_DATABASE_URL` and `psql`) |
| Drift check (`db-change`): `alembic upgrade head` into one database, `build/schema.sql` into another, then `tools/drift.py` on both. `psql` is not installed here, so `drift.py`'s `psql` calls were routed through the Postgres container by a throwaway wrapper in the scratchpad; the repo is unchanged. | `no drift`, exit 0 |
| `mkdocs build --strict` (pinned `docs-requirements.txt` in a scratchpad virtualenv, output outside the repo) | built in 1.3 s, exit 0, no warnings |
| `openspec validate saved-conversations` | Change 'saved-conversations' is valid |

**Not run, so not claimed as passed:**
- The `plumbline-drift` and `datamodel` CI jobs themselves (they are not required checks; look for them green on the PR). The local drift run above used the same two inputs and `drift.py`, but not the CI workflow.
- The eval (none applies; see Eval Results).

## Scenario Coverage

Backend rows run against real tables or the FastAPI test client. Frontend rows are static renders and unit tests (no jsdom).

| Spec scenario | Test |
| --- | --- |
| Conversations are stored: First question creates a conversation | `test_chat_api.py::test_done_carries_answer_id_for_answers_and_abstentions` (`done` carries `answer_id` and `conversation_id`); `test_conversations.py::test_recorder_creates_then_joins` (conversation created and titled from the question) |
| Follow-up joins the conversation | `test_chat_api.py::test_follow_up_joins_the_conversation`; `test_conversations.py::test_recorder_creates_then_joins` (second answer in the same conversation, `updated_at` moves) |
| Unknown conversation | `test_chat_api.py::test_unknown_conversation_is_404_before_answering` (404, `llm.calls == 0`, no recorder call) |
| Failed answer stores nothing | `test_chat_api.py::test_failed_answer_creates_no_conversation`, `test_error_path_records_nothing_and_has_no_answer_id`, `test_recording_failure_sends_error_not_the_answer`; `test_conversations.py::test_recorder_failure_leaves_no_conversation` |
| Conversations survive a reload: Reload restores the chat | Backend: `test_conversations_api.py::test_get_returns_the_turns`, `test_list_is_empty_then_newest_first`. Frontend: `chat/restore.test.tsx` (`turnsToMessages`, a restored thread shows answers, sources and feedback controls), `shell/chatTarget.test.ts` (`syncTarget` on reload). Live reload: see Browser check |
| Own URL | `test_ui_mount.py::test_app_path_serves_index` (`/app/chats/conv_X` serves the index); `shell/route.test.ts` (`chatIdFor`); `shell/AppShell.test.tsx` "opening a saved chat" |
| Header shows title and scope | `chat/ChatHeader.test.tsx`; `shell/AppShell.test.tsx` "chat header" (title from the list, count, "New chat", no header on loading, not-found, error) |
| New chat keeps the previous conversation: New chat | `shell/chatTarget.test.ts` (`newChat`), `shell/AppShell.test.tsx` ("shows an empty chat for New chat"); `test_conversations_api.py::test_list_is_empty_then_newest_first` for the list. The refresh in the UI: see Browser check |
| Rename and delete: Rename | `test_conversations_api.py::test_rename_updates_the_title`; `test_conversations.py::test_rename_trims_and_updates`; `conversations/ChatList.test.tsx` (rename editor with Save and Cancel, submit); `conversations/useConversations.test.tsx` (`renameOutcome`) |
| Invalid title | `test_conversations_api.py::test_rename_rejects_invalid_titles` (422, title unchanged), `test_rename_without_a_title_is_422`; `test_conversations.py::test_rename_rejects_invalid_titles` (raises `InvalidTitle`); `test_migrations.py::test_conversation_title_length` |
| Delete | `test_conversations_api.py::test_delete_then_gone`; `conversations/ChatList.test.tsx` (confirmation and handlers); `shell/chatTarget.test.ts` (`leavesAfterDelete`); `shell/AppShell.test.tsx` ("Conversation not found" for an unknown id) |
| Feedback survives delete | `test_conversations_api.py::test_delete_keeps_the_rated_answer`; `test_conversations.py::test_delete_keeps_answers_and_feedback`; `test_migrations.py::test_deleting_a_conversation_keeps_its_answers` |
| Deleted sources are marked: Cited document deleted | `test_conversations_api.py::test_deleted_source_is_marked_and_the_answer_stays`; `test_conversations.py::test_get_marks_deleted_sources_and_keeps_the_answer`; `chat/Citations.test.tsx` ("no longer indexed"); `chat/restore.test.tsx` (note in a restored thread) |
| Chats section in the sidebar: No conversations | `shell/Sidebar.test.tsx` "Chats section" (hidden when empty or collapsed); `conversations/ChatList.test.tsx` (renders nothing when empty) |
| Accessible list | `shell/a11y.test.tsx` "keyboard reach in the Chats list" (names, links, buttons, one `aria-current`); `conversations/ChatList.test.tsx` (`aria-current`, names, Rename and Delete shown on hover and on focus, bubble icon `aria-hidden`); `shell/a11y.test.tsx` source scans include `ChatList.tsx` and `ChatHeader.tsx`; `design-tokens.test.ts` (header text tokens measured at 4.5:1, light and dark) |
| app-shell: Return to an unchanged chat | `shell/AppShell.test.tsx` ("keeps the chat mounted but hidden on the Documents page"); `shell/chatTarget.test.ts` (target unchanged while the URL names the same chat). Live visit and return: see Browser check |
| app-shell: New chat clears it | as "New chat" above |

Every scenario has at least one automated test. Where a scenario is a live browser flow, the unit-level evidence is listed and the live result is under Browser check.

## Eval Results

N/A per eval-plan: the graph, retrieval and prompt are unchanged; answers are only grouped into conversations after they are produced. The issue's `answer-quality-change` label is therefore satisfied by that statement, and no eval was run.

## Deviations

- **Index direction:** `design.md` originally wrote `conversations (updated_at DESC)`. The datamodel index schema has no sort direction, so `plumbline-drift` would flag it. Revision `0003` uses a plain `(updated_at)` index and `design.md` was updated to match (agreed at task 1.1).
- **Restored citations** carry no `[cN]` id and no quoted text: `answers.citations` keeps only `document_id`, `title`, `filename` and `locator` (design D1, #12). The UI shows `title - locator` with no expander and the "no longer indexed" note (option A, agreed at task 3.2). Inline `[c3]` markers in a restored answer have no matching label.
- **Recorder** replaced the bare recording lambda behind `get_recorder`; it also answers `resolve(conversation_id)`, so the 404 check happens before the graph runs.
- **Delete of the open chat** goes to New chat (decision 2 of task 3.4); the spec is silent on it.
- **Sidebar Chats section hidden when the sidebar is collapsed** (decision 3 of task 3.4); the spec is silent on it.
- **Rename bumps `updated_at`**, so a renamed chat moves to the top of the list; the design is silent on it.
- **Review fixes (commit `703b3b1`):** `rename` raises a domain exception `InvalidTitle` that the route maps to 422 (backend guide 3.1); the duplicate `Citation` type and the unused `loaded` state were removed; the delete-open-chat condition is the tested function `leavesAfterDelete`.
- **Design changes after the first browser run:** each Chats row has a chat bubble icon, Rename and Delete show only on hover or keyboard focus (always on touch screens), and the inline rename editor is a row-shaped input with Save and Cancel icon buttons (commits `cc1a2be`, `ff89878`). The delete confirmation keeps its two text buttons.
- **Outside the issue's task list:** the sidebar Docs link opens the docs site, https://plumbline-docs.netlify.app/ (commit `23b5eaa`), at the request of the person running the check.
- Existing tests edited because the new UI made their markers ambiguous: three `AppShell` tests that used `<h1` as a stand-in for the Documents page now look for `>Documents</h1>` and `>Connectors and quality</h1>`.

## Browser check

The person reviewing ran the UI against the dev servers (backend on :8001, Vite on :5174, Ollama in Docker with `llama3.1:8b` and `nomic-embed-text`) and reported that the functionality works, then asked for the design changes above, which were made and accepted ("lgtm"). This is their report, not an item-by-item record: they did not report dark mode, the drawer below 768 px or the layout-under-header checks one by one. The checklist they were given:

1. New chat to first answer: URL becomes `/app/chats/{id}`, chat appears in the sidebar, header shows the title; reload restores answers, sources and feedback controls.
2. New chat shows an empty thread and the previous chat stays; Documents and back leaves the chat unchanged.
3. Rename (Enter saves, Escape cancels) updates sidebar and header; delete with confirmation, including the open chat, which goes to a new chat; the old URL shows "Conversation not found".
4. A deleted cited document: the answer stays and its source says "Source no longer indexed".
5. Layout, light and dark themes, sidebar drawer, collapsed sidebar.

## Verdict

PASS. Every check above passed on `ff89878`, and the browser check is the reviewer's report as described.

**Before merge:** look for the `plumbline-drift` and `datamodel` CI jobs green on the PR (not required checks). If anyone wants the dark-mode and narrow-width items recorded per item, they are the open part of the Browser check.
