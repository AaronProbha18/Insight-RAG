## 1. Backend: answers and the stream

- [x] 1.1 Migration `0002_answers_feedback.py` (tables, checks, downgrade). Test: `test_migrations.py` - upgrade/downgrade round trip; `CHECK`s reject a reason on `up` and an unknown reason.
- [x] 1.2 `record_answer()` in `plumbline/feedback.py`; carry `filename` on retrieved chunks/citations internally if absent; `/chat` records answers and abstentions and sends `done` with `answer_id`; error path sends none. Test: `test_chat_api.py` - `done.answer_id` starts `ans_` and the row holds question, answer, citations (id, title, filename, locator), model ids, threshold; abstention recorded with `abstained=true`; error path has no `answer_id`; `filename` never appears in the `citations` event.

## 2. Backend: feedback API and export

- [x] 2.1 `POST /feedback` with `save_feedback()` upsert: 204, 422 (bad rating/reason, reason on `up`, comment > 2000), 404 "Answer not found". Test: `test_feedback_api.py` - each AC case incl. abstention + `should_have_answered`, change replaces (one row), no internal details, `test_no_confidence_field_anywhere` still passes.
- [x] 2.2 `plumbline-feedback-export --out PATH` (console script, default `evals/feedback/candidates.yaml`, git-ignored). Test: `test_feedback_export.py` - the four mapping rows of the design table; `reference_answer` empty; output loads as YAML list in golden shape; golden files untouched; works after a cited document is deleted.

## 3. Frontend

- [ ] 3.1 `adapter.ts` reads `answer_id` from `done` into `metadata.custom.answerId`; `feedback/api.ts` `sendFeedback()`. Test: `adapter.test.ts` - `answerId` set for answers and abstentions, absent on error; `api.test.ts` stubbed `fetch` - 204 saved, 404 not-found, 422/500/network throw.
- [ ] 3.2 Amend ADR 0003; vendor the kit feedback part into the assistant action bar (or plain buttons in kit styling if the UNVERIFIED API is missing); thumbs-down reason form; recorded state shown; hidden while running or on error. Test: static markup with `ReadonlyThreadProvider` - controls with accessible names on a finished answer and an abstention, none on a running message; `npm run lint && npm run build && npm test`; keyboard, focus, form and both themes checked by hand.

## 4. Docs

- [ ] 4.1 Backend guide (status codes, D8 `done` payload), frontend guide (feedback controls, manual checks), architecture overview (answers/feedback tables, what is stored locally), `evals/README.md` (export command and review flow). Test: `python scripts/check_links.py` passes.
