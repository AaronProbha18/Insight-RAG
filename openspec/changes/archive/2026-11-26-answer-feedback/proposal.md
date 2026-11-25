Issue: #13

## Why

When an answer is wrong, cites the wrong source, or the assistant refuses
a question its documents cover, people can't say so, and maintainers get
no signal about which real questions fail. Golden sets only cover
synthetic questions.

## What Changes

- Every completed answer (abstentions included) is recorded in a new
  local `answers` table: question, answer text, abstained flag, citations
  (document id, title, filename, locator), chat and embedding model ids,
  abstain threshold.
- **Stream contract:** the `/chat` `done` event carries a public
  `answer_id` (`ans_<ULID>`); `frontend/src/chat/adapter.ts` changes in
  the same diff.
- New `POST /feedback`: thumbs up or down, optional reason on thumbs down
  (wrong, incomplete, wrong source, should have answered, other) and
  optional comment. One record per answer; a later value replaces the
  earlier one. 204 on success, 422 invalid, 404 unknown `answer_id`.
- Chat UI: thumbs up / down on each completed assistant message; thumbs
  down opens an optional reason + comment form; the control shows the
  recorded value.
- New `plumbline-feedback-export --out PATH` writes every feedback record
  as golden-question candidates in the `golden.yaml` shape, with
  `reference_answer` empty. Nothing is added to golden sets automatically.
- ADR 0003 amended: the styled kit's feedback part is brought back; no
  other trimmed parts.

## Non-Goals

- Analytics dashboard or clustering failed questions (Gap Finder).
- Editing golden sets automatically.
- User identity or accounts.
- Any other trimmed kit parts (tool calls, attachments, branches).
- Retention limits on the answers table.

## Capabilities

### New

- `answer-feedback`: recording answers, collecting feedback on them, and
  exporting it as golden-question candidates.

### Modified

- `chat-ui`: assistant messages gain feedback controls.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Answers are recorded after the graph runs; retrieval, prompts and the graph are unchanged. Feedback is a signal for future evals, not an input to answering. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | No identity (non-goal); anyone with the UI can give feedback, as anyone can ask. |
| Public API or stream contract | Yes | `done` event gains `answer_id`; new `POST /feedback`; new console script. |
| Persistence / migration | Yes | Migration `0002` adds `answers` and `feedback`. Every answer is stored locally, with no retention limit. |
| New external dependency or service | No | Kit feedback part is vendored source; no new package (UNVERIFIED - see design). |

## Impact

`backend/src/plumbline/api/app.py`, new `backend/src/plumbline/feedback.py`,
`backend/migrations/versions/0002_*.py`, `backend/pyproject.toml` (script),
backend tests; `frontend/src/chat/adapter.ts`, `thread.aui.tsx`, new
feedback component, frontend tests; `docs/adr/0003-styled-chat-kit.md`,
backend and frontend guides, architecture overview, `evals/README.md`,
`openspec/specs/chat-ui`.
