## Checks Run

All run in this session. Backend DB tests ran against a throwaway `pgvector/pgvector:pg18` container on port 55432 (the dev DB container rejects the `plumbline` credentials, probably created before the rename), `DATABASE_URL` pointed at it. Frontend commands run from `D:\` (upper-case).

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | All checks passed |
| `cd backend && uv run ruff format --check .` | 58 files already formatted |
| `cd backend && uv run pytest -q -rs` | 178 passed, 1 skipped (`test_models.py:66`, needs Ollama; unrelated) |
| `cd frontend && npm run lint` | oxlint clean |
| `cd frontend && npm run build` | `tsc -b` + Vite build OK |
| `cd frontend && npm test` | 57 passed, 15 files |
| `python scripts/check_links.py` | links ok (81 files) |
| `openspec validate answer-feedback` | valid |
| `uv run plumbline-feedback-export --out <tmp>` (task 2.2 smoke, earlier this change) | wrote the expected YAML |

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| answer-feedback: done carries an answer id | `test_chat_api.py::test_done_carries_answer_id_for_answers_and_abstentions`, `test_recorder_gets_question_answer_and_filename`; `test_feedback.py::test_answer_row_holds_everything_needed_to_review_it`; `adapter.test.ts` (answerId set for answers and abstentions) |
| Thumbs up | `test_feedback_api.py::test_thumbs_up`; `api.test.ts` (204 -> saved); `FeedbackControls.test.tsx` saved state shows "Feedback recorded" and the pressed button. The click path itself is not exercised (see Deviations) |
| Thumbs down with reason and comment | `test_feedback_api.py::test_thumbs_down_keeps_reason_comment_and_the_answer_record`; `FeedbackControls.test.tsx` choosing state |
| Feedback on an abstention | `test_feedback_api.py::test_feedback_on_an_abstention`; `thread.aui.test.tsx` (controls on an abstention) |
| Changing feedback | `test_feedback_api.py::test_changing_feedback_replaces_it` |
| Invalid or unknown | `test_feedback_api.py::test_invalid_feedback_422_stores_nothing`, `test_unknown_answer_404_reveals_nothing`; `test_chat_api.py::test_no_confidence_field_anywhere` |
| Export | `test_feedback_export.py` (mapping rows, golden-shaped YAML, golden files untouched, deleted document, empty) |
| chat-ui: Keyboard only | Accessible names: `thread.aui.test.tsx`, `FeedbackControls.test.tsx`. Contrast: `design-tokens.test.ts` (existing tokens only). Keyboard operation, visible focus rings and both themes: **manual check, reported by the maintainer** (see Manual Checks) |
| chat-ui: No controls while streaming or on errors | `thread.aui.test.tsx::offers no feedback while running or without an answer id`; `adapter.test.ts::yields no answerId before an error` |

## Manual Checks

Done by the maintainer in a browser against a stack started by Claude (Vite dev server, uvicorn with a stub LLM and retriever, throwaway Postgres). Claude cannot drive a browser, so this is the maintainer's report, not Claude's observation. Reported as passing: the feedback buttons under answers work; keyboard-only use (Tab reaches Good and Bad answer, Enter and Space press them, arrow keys across the reasons, Cancel and Send return focus to Bad answer); visible focus rings; light and dark themes. Not itemised in the report: how the repeated final update renders.

## Eval Results

N/A per eval-plan: answers are recorded after the graph runs; retrieval, prompts and the graph are unchanged. The issue has no `answer-quality` label.

## Deviations

- **Kit feedback primitives not used (ADR 0003 amended).** `FeedbackAdapter.submit` in `@assistant-ui/react` 0.15.37 returns `void`, carries no reason and cannot report a failed save, so the feedback controls are our own component built from the kit's buttons. design.md named this as the fallback.
- **Recording failure returns `error`**, not an unrecorded answer (decided in plan 1.2, approved).
- **Default FastAPI 422 body kept** (echoes the sent input and field path, no internals), as decided in plan 2.1.
- **Dev DB container** `insight-rag-v2-db-1` rejects the `plumbline` credentials, so DB tests do not run against it; recreate it with `docker compose down -v && docker compose up -d db`.
- **Dev proxy gap found and fixed during the manual pass:** `vite.config.ts` did not forward `/feedback`, so ratings never reached the backend in dev (production unaffected: same origin). Fixed in `138a276`; through the proxy a thumbs-down returned 204 and was stored, and an unknown id returned 404.

## Verdict

PASS. Every automated check passes, every spec scenario maps to a test or a recorded manual check, and the eval plan is N/A. Caveats: the keyboard, focus-ring and theme evidence is the maintainer's report, not independently observed; the dev DB container still needs recreating locally (`docker compose down -v && docker compose up -d db`); the automated checks above were not re-run after the one-line `vite.config.ts` fix.
