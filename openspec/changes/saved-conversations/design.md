## Context

`POST /chat` answers one question (the last message) and records it as one `answers` row, with citations copied at answer time (#12), then streams `delta…`, `citations`, `done{answer_id}` or `abstain`, `done`. The thread lives in assistant-ui's local runtime, so a reload or New chat loses it. There are no accounts. Tables use a `bigserial` key internally and a prefixed `public_id` externally.

## Decisions

### D1. Turns are `answers` rows

- **Choice**: a conversation is a `conversations` row; its turns are the `answers` rows pointing at it, in `created_at` order.
- **Alternatives**: a separate `messages` table with role and content.
- **Why**: `answers` already holds the question, the answer, the abstention flag and the copied citations; a second table would duplicate them.
- **ADR**: none (local to this change).

### D2. Created on the first question

- **Choice**: `POST /chat` without `conversation_id` creates the conversation in the same transaction as the answer. New chat only resets the UI to `/`.
- **Alternatives**: create on New chat.
- **Why**: empty conversations are never stored, so "no conversations hides Chats" needs no cleanup. A failed answer is not recorded (existing rule), so it creates no conversation either.
- **ADR**: none.

### D3. Delete keeps answers

- **Choice**: `answers.conversation_id` is `ON DELETE SET NULL`.
- **Alternatives**: `CASCADE`, which also deletes feedback through `answers`.
- **Why**: feedback feeds `plumbline-feedback-export`; deleting a chat must not destroy rated answers.
- **ADR**: none.

### D4. The server marks deleted sources

- **Choice**: `GET /conversations/{id}` sets `indexed` on each citation, `true` when its `document_id` still exists in `documents`.
- **Alternatives**: the frontend compares against `GET /documents`.
- **Why**: one query, and the UI needs no second fetch or list in scope.
- **ADR**: none.

### D5. Title

- **Choice**: the first question, whitespace collapsed, cut to 80 characters (with "…" when cut). A rename is trimmed and must be 1–200 characters, else 422.
- **Alternatives**: an LLM-generated title (changes answering cost and is a non-goal).
- **ADR**: none.

### D6. No history to the model

- **Choice**: `/chat` still answers only the last message.
- **Why**: issue non-goal; answer quality stays unchanged, so no eval applies.
- **ADR**: none.

### D7. Frontend restore

- **Choice**: `/app/chats/{id}` loads `GET /conversations/{id}` and passes the turns to the local runtime as initial messages, with citations and `answer_id` in message metadata, so the existing citations and feedback render. Subsequent questions send `conversation_id`. After the first `done` in a new chat, the URL is replaced with `/app/chats/{id}` and the list is refreshed.
- **Alternatives**: assistant-ui's thread-list / external-store runtime.
- **Why**: fewer moving parts; the local runtime is already in use.
- UNVERIFIED - confirm by a unit test in task 3.2 that `useLocalRuntime` accepts `initialMessages` with the metadata shape the existing citation and feedback components read.
- **ADR**: none.

## Interfaces

```python
class ChatRequest(BaseModel):
    messages: list[Message]
    conversation_id: str | None = None   # "conv_…"; unknown -> 404 before answering

# SSE done event
{"answer_id": "ans_…", "conversation_id": "conv_…"}

# GET /conversations -> 200
[{"conversation_id": "conv_…", "title": str, "updated_at": iso8601}]   # newest updated first

# GET /conversations/{id} -> 200 | 404 "Conversation not found"
{"conversation_id": "conv_…", "title": str,
 "turns": [{"answer_id": "ans_…", "question": str, "answer": str, "abstained": bool,
            "citations": [{"document_id", "title", "locator", "indexed": bool}]}]}

# PATCH /conversations/{id} {"title": str}  -> 200 {conversation_id, title, updated_at} | 404 | 422
# DELETE /conversations/{id}               -> 204 | 404
```

```python
# conversations.py
def create(conn, first_question: str) -> tuple[int, str]: ...      # (id, public_id)
def resolve(conn, public_id: str) -> int | None: ...
def touch(conn, conv_id: int) -> None: ...                           # updated_at = now()
def list_all(conn) -> list[dict]: ...
def get(conn, public_id: str) -> dict | None: ...
def rename(conn, public_id: str, title: str) -> dict | None: ...
def delete(conn, public_id: str) -> bool: ...
```

## Data

Revision `0003`:

```sql
CREATE TABLE conversations (
  id          bigserial PRIMARY KEY,
  public_id   text UNIQUE NOT NULL,
  title       text NOT NULL CHECK (char_length(title) BETWEEN 1 AND 200),
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversations_updated_at ON conversations (updated_at DESC);
ALTER TABLE answers ADD COLUMN conversation_id bigint
  REFERENCES conversations(id) ON DELETE SET NULL;
CREATE INDEX answers_conversation_id ON answers (conversation_id);
```

Downgrade drops the column, then the table. Existing answers keep `conversation_id = NULL` and are not listed. Mirror in `datamodel/models/plumbline` (ADR 0004): new `conversations.yaml` (prefix `conv`), `answers.yaml` gains `conversation_id`.

## Risks / Open Questions

- Free-text titles share `answers` OQ-ANSWERS-001 (personal details typed into questions; retention). Reference it from `conversations.yaml`; no new decision here.
- Single-user assumption: anyone who can reach the app can list every conversation, as with documents today. Revisit when accounts exist.
