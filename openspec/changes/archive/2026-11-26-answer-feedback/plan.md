## Task
4.1 Backend guide (status codes, D8 `done` payload), frontend guide (feedback controls, manual checks), architecture overview (answers/feedback tables, what is stored locally), `evals/README.md` (export command and review flow). Test: `python scripts/check_links.py` passes.

## Scope
Docs only, each edit states what the code now does (read from the committed code, not the design):
- `docs/guides/backend-development-guide.md`:
  - §2.1 status table: `POST /feedback` rows (204 stored or replaced, 404 "Answer not found", 422 invalid body incl. reason on thumbs up or comment over 2000).
  - §2.2/§2.4: `done` now carries `answer_id` (`ans_<ULID>`, absent on the error path); `filename` stays internal and is stripped from `citations`; an answer that can't be recorded is not streamed (`error` instead).
  - §1.3 or §8: note the `get_recorder` dependency as the test seam (fake recorder in `test_chat_api.py`).
- `docs/guides/frontend-development-guide.md`: §2.2 `answerId` in `metadata.custom` (set on the final update after `done`); §4.1 feedback is our own `feedback/FeedbackControls.tsx` (ADR 0003 amendment), the other trimmed parts stay out; §5 accessibility: control names ("Good answer", "Bad answer", "Feedback recorded" status) and the reason form's native fieldset/radios/label; §7.1 manual-check list for feedback (Tab order, Enter/Space, radio arrows, Cancel returns focus, focus rings, both themes) and static-test coverage.
- `docs/architecture/overview.md`: diagram edge `POST /feedback`; "Where things live" row (`feedback.py`, console script); Ask flow step 5 (`answer_id`, recording, failure behaviour); new "Feedback: `POST /feedback`" and "Export: `plumbline-feedback-export`" flows; data model rows for `answers` and `feedback` (ids, `ON DELETE CASCADE`, one feedback row per answer, citations copied as JSONB, why); a **what is stored locally** note (every question and answer is kept with no retention limit; no real personal data).
- `evals/README.md`: new section "Feedback candidates": the command, default output, the mapping table, what a reviewer must fill (`reference_answer`, `key_fact`, `answerable: null` cases), delete the `feedback:` block, copy into a connector's `golden.yaml` by hand; the file holds real questions, is git-ignored, never commit it; golden sets are never modified automatically.
- `README.md`: one line next to "synthetic data only" that questions and answers are stored locally.
- Update the Mermaid/diagram and tables only where the behaviour changed (surgical).

## Not in scope
- Changing code, specs, ADR (done in 3.2), or archived rag-foundation design.
- Docs for features other than #13.

## Files
| File | Change |
| --- | --- |
| docs/guides/backend-development-guide.md | status table, `done` payload, test seam |
| docs/guides/frontend-development-guide.md | `answerId`, feedback component, a11y, manual checks |
| docs/architecture/overview.md | diagram, flows, data model, what is stored |
| evals/README.md | feedback candidates section |
| README.md | one line on stored questions |

## Tests
- `python scripts/check_links.py` passes (all new links resolve).
- Claims checked against code: grep each named symbol/file/command in the docs exists (`get_recorder`, `FeedbackControls`, `plumbline-feedback-export`, `answers`/`feedback` columns).
- Existing doc-linked tests untouched (`test_check_links.py`, `test_old_names.py`: no pre-rename names added).
- Backend and frontend suites not re-run (no code change); I'll run `uv run pytest tests/test_old_names.py tests/test_check_links.py -q` since they scan docs.

## Checks to run
- `python scripts/check_links.py`
- `cd backend && uv run pytest tests/test_old_names.py tests/test_check_links.py -q`

## Risks / Questions
- The guides claim some things are "enforced by" a test; I will cite only tests that exist (`test_chat_api.py`, `test_feedback_api.py`, `test_feedback_export.py`, `thread.aui.test.tsx`, `FeedbackControls.test.tsx`).
- The manual checks I wrote in 3.2 are still unrun; the frontend guide will list them as manual steps, not as done.
