## 1. Backend

- [x] 1.1 `backend/src/plumbline/unanswered.py` `list_groups(conn, days)`: the grouped query (design D1-D4). Tests against the test database: case/space grouping with count and latest date, "marked wrong" with `feedback_reason`, the latest event's reason wins, 40 days ago excluded from 30, all time when `days` is None, empty list.
- [x] 1.2 `GET /unanswered` route in `api/app.py`: accepts `days` in {7, 30, 90} or absent (else 422) and calls `list_groups`. API tests: 200 shape, 422 on `days=5`, no scores, ids or answer text in the body.

## 2. Frontend

- [x] 2.1 `frontend/src/unanswered/api.ts` and `useUnanswered.ts` (fetch with `days`, `refresh()`, as `useConversations` does), plus the `/app/unanswered` page in `shell/route.ts`. Tests: fetch URL per range, refresh refetches, the route maps the path.
- [x] 2.2 `UnansweredPage.tsx`: rows (question, "Asked N times", last date, reason text using the `FeedbackControls` labels, Add a document link to `/app/documents`), range filter defaulting to Last 30 days, empty state. Tests for each, plus a keyboard/a11y test in the style of `shell/a11y.test.tsx`.
- [x] 2.3 Sidebar **Unanswered** item showing the last-30-days group count, wired through `App.tsx` and `AppShell`; `App.tsx` refreshes it after each chat answer next to `conversations.refresh()`. Tests: the item shows the count, and the count refreshes after an answer.
- [x] 2.4 Refused assistant messages show an "Added to Unanswered" link to `/app/unanswered` (find the rendering component first, per design). Test: an abstained message has the link, an answered one does not.

## 3. Docs

- [x] 3.1 `docs/architecture/overview.md`: add `GET /unanswered` to the diagram, the "Where things live" table and a short flow. Check: the docs build passes (`docs/guides/docs-site.md`).
