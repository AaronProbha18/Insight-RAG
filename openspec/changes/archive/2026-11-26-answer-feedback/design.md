## Context

`/chat` runs the graph to completion, then streams `delta`, `citations`
(or `abstain`) and an empty `done` (design D8 of `rag-foundation`). It has
no DB connection today. Citations carry `document_id`, `title` and
`locator`. Golden sets (`evals/connectors/*/golden.yaml`) name documents
by filename stem (`employee-handbook`) plus locator, not by `doc_` id, and
documents can now be deleted (#12). The kit's feedback part was trimmed
by ADR 0003. Model ids come from `PLUMBLINE_LLM` / `PLUMBLINE_EMBEDDINGS`;
the threshold from `PLUMBLINE_ABSTAIN_THRESHOLD`.

## Decisions

### Record every answer at `done`

- **Choice**: after the graph returns (answer or abstain), insert one
  `answers` row and put its `public_id` in `done`. Errors record nothing
  and send no `answer_id`.
- **Alternatives**: client sends question/answer with the feedback (can't
  404 an unknown id, trusts client text); record only when feedback
  arrives (same problem); prune answers without feedback after N days.
- **Why**: AC needs 404 for unknown ids and server-side truth for what
  was answered. Product owner chose to keep every answer, no pruning.
- **ADR**: none (local to this change)

### Citations are copied, not joined

- **Choice**: `answers.citations` is JSONB holding `document_id`, `title`,
  `filename`, `locator` per citation, captured at answer time.
- **Alternatives**: join to `documents` at export time.
- **Why**: a cited document may be deleted later (#12); the export still
  needs the filename stem for `supporting`.
- **ADR**: none

### One feedback row per answer, upserted

- **Choice**: `feedback.answer_id` UNIQUE; `INSERT ... ON CONFLICT
  (answer_id) DO UPDATE`. Reason only allowed with `down`; comment max
  2000 chars. Validation in Pydantic (422) and DB `CHECK`s.
- **Alternatives**: append history, read latest.
- **Why**: AC "one record per answer"; history isn't asked for.
- **ADR**: none

### Export everything as candidates

- **Choice**: `plumbline-feedback-export --out PATH` (default
  `evals/feedback/candidates.yaml`, git-ignored) writes one entry per
  feedback record:

  | Feedback | `answerable` | `supporting` |
  | --- | --- | --- |
  | up on an answer | `true` | cited filename stem + locator |
  | up on an abstention | `false` | `[]` |
  | down, reason `should_have_answered` | `true` | `[]` |
  | down, any other reason or none | `null` (reviewer decides) | cited filename stem + locator |

  `id: fb-<answer_id>`, `reference_answer: ""`, `key_fact: ""`, plus a
  `feedback:` block (rating, reason, comment, answer text, model ids,
  threshold) for the reviewer to read and delete. Needs only the DB, no
  models.
- **Alternatives**: thumbs-down only; a flag on `plumbline-eval`.
- **Why**: product owner chose everything and a separate command.
- **ADR**: none

### Bring back the kit feedback part only

- **Choice**: amend ADR 0003 (Consequences + a dated amendment line) to
  vendor the feedback buttons into `thread.aui.tsx`'s action bar, wired
  to our own `POST /feedback` client; a small reason form (radio group +
  textarea) opens on thumbs down.
- **Alternatives**: hand-built buttons outside the kit.
- **Why**: issue asks for the kit part; keeps styling consistent.
- **ADR**: docs/adr/0003-styled-chat-kit.md (amended)
- UNVERIFIED - confirm `ActionBarPrimitive.FeedbackPositive/Negative` and
  the runtime `feedback` adapter exist in the installed
  `@assistant-ui/react`, and whether the adapter's submit can carry
  `answer_id` from `metadata.custom`; if not, use plain buttons inside the
  kit's action bar styling.

## Interfaces

```python
# SSE (contract change, design D8 amended)
event: done
data: {"answer_id": "ans_01J..."}      # absent only on the error path

# POST /feedback
class FeedbackRequest(BaseModel):
    answer_id: str
    rating: Literal["up", "down"]
    reason: Literal["wrong", "incomplete", "wrong_source",
                    "should_have_answered", "other"] | None = None  # only with "down"
    comment: str | None = Field(None, max_length=2000)
# 204 stored | 404 {"detail": "Answer not found"} | 422 validation

# plumbline/feedback.py
def record_answer(conn, question: str, result: dict, *, chat_model: str,
                  embedding_model: str, threshold: float) -> str: ...  # returns ans_ id
def save_feedback(conn, req: FeedbackRequest) -> bool: ...             # False -> 404
def export_candidates(conn) -> list[dict]: ...
def main() -> None: ...                                                 # plumbline-feedback-export
```

```ts
// frontend/src/chat/adapter.ts: metadata.custom gains answerId
yield { content, metadata: { custom: { citations, abstained, answerId } } }
// frontend/src/feedback/api.ts
export async function sendFeedback(f: Feedback): Promise<'saved' | 'not-found'>
```

## Data

Migration `0002_answers_feedback.py`:

```sql
CREATE TABLE answers (
  id                bigserial PRIMARY KEY,
  public_id         text UNIQUE NOT NULL,          -- ans_<ULID>
  question          text NOT NULL,
  answer            text NOT NULL,
  abstained         boolean NOT NULL,
  citations         jsonb NOT NULL,                -- [{document_id,title,filename,locator}]
  chat_model        text NOT NULL,
  embedding_model   text NOT NULL,
  abstain_threshold double precision NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE feedback (
  id          bigserial PRIMARY KEY,
  answer_id   bigint UNIQUE NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
  rating      text NOT NULL CHECK (rating IN ('up','down')),
  reason      text CHECK (reason IN ('wrong','incomplete','wrong_source','should_have_answered','other')),
  comment     text CHECK (char_length(comment) <= 2000),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  CHECK (reason IS NULL OR rating = 'down')
);
```

Downgrade drops both. `filename` for citations requires retrieval's
chunk record to carry it - add `filename` to `RetrievedChunk` if absent
(internal only; not sent to clients).

## Risks / Open Questions

- Every question asked is stored locally with no retention limit. Users
  must not paste real personal data; README and overview say what is
  stored.
- `done` is a contract change: adapter and backend change in one PR
  (backend guide §2.4).
- Static-markup tests can't click; keyboard, focus and the reason form are
  checked by hand in both themes.
