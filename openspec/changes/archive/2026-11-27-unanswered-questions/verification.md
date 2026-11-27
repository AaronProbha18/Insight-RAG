## Checks Run

All run in this session on branch `feat/37-unanswered-questions` (HEAD `2f7d56e`). Postgres was up (`docker compose up -d db`), so DB tests ran.

| Command | Result |
| --- | --- |
| `openspec status --change unanswered-questions` / `tasks.md` | 7 of 7 checkboxes `[x]`, 0 open |
| `openspec validate unanswered-questions` | "Change 'unanswered-questions' is valid" |
| `cd backend && python -m ruff check .` | All checks passed |
| `cd backend && python -m ruff format --check .` | 69 files already formatted |
| `cd backend && python -m pytest -q -rs` | 266 passed, 1 skipped (`test_models.py:66`, needs Ollama; not related) |
| `cd frontend && npm run lint` (oxlint) | no warnings or errors |
| `cd frontend && npm run build` (`tsc -b` + Vite) | built, no errors |
| `cd frontend && npm test` (run from `D:\`-style path via `/D/…`) | 39 files, 312 tests passed |
| `mkdocs build --strict` (docs venv in scratchpad, `docs-requirements.txt`) | built, no warnings |
| drift check | Not required: `git diff origin/main...HEAD` touches nothing under `backend/migrations` or `datamodel`, and the issue has no `db-change` label |

Note: `uv run` fails on this machine ("uv trampoline failed to canonicalize script path"), so Python checks ran as `python -m <tool>`; the project's `uv run` form was not run.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Same question grouped | `backend/tests/test_unanswered.py::test_case_and_spacing_group_into_one` (count 3, latest text and date); `frontend/src/unanswered/UnansweredPage.test.tsx` "shows the question, count, date and reason of a row" ("Asked 3 times", "Asked 1 time") |
| Marked wrong with a reason | `test_unanswered.py::test_marked_wrong_carries_the_reason`, `::test_latest_event_reason_wins`; `UnansweredPage.test.tsx` "words a wrong-marked reason as text" ("Marked wrong: wrong source", plain "Marked wrong") |
| Outside the range | `test_unanswered.py::test_range_filters_events_before_counting` (40 days out of 30, in 90 and all time); `test_unanswered_api.py::test_days_filters_the_range` |
| Invalid range | `test_unanswered_api.py::test_invalid_days_is_422` (`days=5`, `days=abc`). Gap, minor: the test asserts the status only, not that the body carries no internal details. The route's own message is "days must be 7, 30 or 90". |
| Empty state | `UnansweredPage.test.tsx` "shows the empty state only when there are no rows" |
| Add a document | `UnansweredPage.test.tsx` "links every row to the Documents page" and "handles a plain click as navigation and leaves modified clicks alone" |
| Keyboard, themes and contrast | Static only: `frontend/src/shell/a11y.test.tsx` (`unanswered` page in the matrix: every control named; source scans for no bare `outline-none` and tokens only, including `UnansweredPage.tsx`). Tab order, light/dark themes and contrast are NOT covered by an automated test (no jsdom, guide 7.1); see Deviations. |
| Refusal links and the count updates | Link: `chat/AddedToUnanswered.test.tsx`, `thread.aui.test.tsx` "links a recorded refusal to Unanswered, and an answer does not". Count: `shell/Sidebar.test.tsx` "links Unanswered with its count…", `chat/adapter.test.ts` "calls onAnswer on every done…". The end-to-end refresh (`App` -> `ChatPane` -> adapter -> `useUnanswered.refresh`) has no rendered test. |

Also covered, not a scenario: `GET /unanswered` shape and no leaks (`test_unanswered_api.py::test_shape_and_no_internals`); `/app/unanswered` route (`shell/route.test.ts`); fetch URL per range (`unanswered/api.test.ts`).

## Eval Results

N/A per eval-plan: it reads stored answers and feedback only; retrieval, prompts, the graph and refusal logic are unchanged. The issue has no `answer-quality-change` label.

## Deviations

- **D5 refresh:** the sidebar count refreshes through a new `ConversationRef.onAnswer` callback (fired on every `done`, threaded `App` -> `AppShell` -> `ChatPane` -> adapter), not "next to `conversations.refresh()`", which only fires on a conversation's first answer. Decided with the user in task 2.3.
- **Route validation:** `days` is `int | None` with a manual 422, not `Literal[7, 30, 90]`; FastAPI did not coerce a Literal of ints from the query string (found by test in task 1.2).
- **Vite proxy:** `/unanswered` added to `frontend/vite.config.ts` (guide 3.1), not listed in the tasks.
- **`REASONS` moved** to `frontend/src/feedback/reasons.ts` so the Unanswered page can reuse it without a lint warning (fast refresh).
- **Restyle after task 2.2:** the page now follows the "App Shell with Sidebar" design artboard (layout, spacing, radii, type scale, our colour tokens). Differences from the design: both reasons use one neutral pill, the date-range select has an `sr-only` label, "Asked N times" kept per spec, and the sidebar was left as-is (no Insights group, "Unanswered (n)"). The page heading is now "Unanswered questions".
- **Known behaviour gaps, out of scope per the proposal:** no dismiss or auto-close (the user will track it), the empty state shows briefly before the first response, and the count does not refresh after a thumbs-down.
- **Manual checks:** the user reviewed the running app with a checklist and reported "LGTM" before the design restyle. The restyle was not re-checked by hand, and Tab order, theme contrast of the new pill and button styles, and the live count refresh on a second chat message have no automated coverage.
- Local `main` is behind `origin/main`; the diff above is against `origin/main`.

## Verdict

PASS
