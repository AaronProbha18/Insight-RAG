## Context

Every `/chat` result is already an `answers` row (`question`, `abstained`, `created_at`), written by `feedback.record_answer`. Thumbs down is a `feedback` row (`rating = 'down'`, optional `reason` in `wrong | incomplete | wrong_source | should_have_answered | other`), one per answer. Question text never leaves the local Postgres. The frontend has no data library: hooks fetch with a version counter and expose `refresh()` (`conversations/useConversations.ts`); routing is `shell/route.ts`; `App.tsx` already calls `conversations.refresh()` after each chat answer.

## Decisions

### D1. Derive the list from existing rows, no new table

- **Choice**: a read-only query over `answers LEFT JOIN feedback`; an event is `abstained OR rating = 'down'`.
- **Alternatives**: a new `unanswered` table written from `/chat` and `/feedback`.
- **Why**: the data already exists; a second copy needs a migration, a write on the chat path and drift handling. Add an index if the query gets slow.
- **ADR**: none (local to this change)

### D2. Grouping key in SQL

- **Choice**: `lower(regexp_replace(btrim(question), '\s+', ' ', 'g'))`. The row shows the latest event's original question text.
- **Alternatives**: normalising in Python after fetching every row.
- **Why**: one query, and it matches the issue's rule exactly (case and extra spaces only).
- **ADR**: none

### D3. The range filters events, then groups

- **Choice**: `days` in {7, 30, 90} filters on `answers.created_at >= now() - days`; omitted means all time; any other value is 422. Count and last date cover in-range events only.
- **Alternatives**: filter groups by last date but count all-time events.
- **Why**: "Asked N times" then agrees with the chosen range, and a question last asked 40 days ago drops out of Last 30 days.
- **ADR**: none

### D4. One reason per row: the latest event's

- **Choice**: `reason` is `not_in_documents` when the latest event is an abstention, else `marked_wrong` with `feedback_reason` (the stored enum, or null). The UI maps enum values to the labels `FeedbackControls` already uses ("Marked wrong: wrong source").
- **Alternatives**: list every reason in a group.
- **Why**: the wireframe shows one reason, and the latest is the most actionable.
- **ADR**: none

### D5. The sidebar count reuses the list

- **Choice**: the sidebar shows the number of groups from `GET /unanswered?days=30`; `App.tsx` calls its `refresh()` next to `conversations.refresh()` after each chat answer.
- **Alternatives**: a separate count endpoint.
- **Why**: one endpoint; the list is small.
- **ADR**: none

## Interfaces

```python
# backend/src/plumbline/unanswered.py
def list_groups(conn: psycopg.Connection, days: int | None) -> list[dict]:
    """Groups, newest last_asked_at first."""

# GET /unanswered?days=30  -> 200
# {"items": [{"question": "What is the refund window?",
#             "count": 3,
#             "last_asked_at": "2026-11-20T10:00:00Z",
#             "reason": "not_in_documents" | "marked_wrong",
#             "feedback_reason": "wrong_source" | null}]}
# days not in {7, 30, 90} -> 422. No scores, ids or answer text in the response.
```

Frontend: `Page` gains `'unanswered'` at `/app/unanswered`; new `frontend/src/unanswered/{api.ts,useUnanswered.ts,UnansweredPage.tsx}`.

## Data

None. Reads `answers` and `feedback` as they are.

## Risks / Open Questions

- Answers whose conversation was deleted still count (they stay in `answers`). That is intended: the signal is about missing documents, not chats.
- Closing a group is out of scope (triage decision), so every in-range group counts as open.
- UNVERIFIED: which component renders an abstained assistant message, for the "Added to Unanswered" link. Confirm by reading the thread message component when planning task 2.4.
