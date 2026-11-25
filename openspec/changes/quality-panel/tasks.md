## 0. Prerequisite

- [x] 0.1 Bring PR #19 into this branch: rebase the stack onto `main` once #15/#17/#18 are merged (or rebase this branch onto `main` with #19), resolving `backend/src/plumbline/evals/runner.py` to the Plumbline names. Test: backend suite passes; a sampled `plumbline-eval` run writes `<report>.summary.json`.

## 1. Backend

- [x] 1.1 `plumbline/quality.py` `load_quality(path, running)` with fixture summaries under `backend/tests/fixtures/quality/` (both pass, docx fails faithfulness, exit 2, sampled, missing field, malformed). Test: `test_quality.py` - state, result, per-connector `failed`, three gated metrics only, `config_matches` and `config_differences` for chat / embeddings / threshold, judge not compared; no `host`, `git_commit`, `report` or path-like value in any output.
- [ ] 1.2 `GET /quality` reading `PLUMBLINE_EVAL_SUMMARY` (default `evals/reports/latest.summary.json`); add it to `.env.example`. Test: `test_quality_api.py` - missing file -> `not_measured` with `eval_command`; fixture -> allowlisted fields only; `test_no_confidence_field_anywhere` and existing `/chat` tests unchanged.

## 2. Frontend

- [ ] 2.1 `frontend/src/quality/api.ts` `getQuality()`. Test: `api.test.ts` stubbed `fetch` - measured and not-measured parse; 500 and network throw.
- [ ] 2.2 `QualityPanel` component: per connector PASS/FAIL as text and icon, three metrics, failed metrics named "below 0.80", run date and model ids, mismatch warning, not-measured state with the command. Test: `QualityPanel.test.tsx` static markup - each AC case; PASS/FAIL words present (not colour alone); no numbers when not measured.
- [ ] 2.3 "Quality: <state>" button in the upload bar opening a `Dialog` with `QualityPanel`; fetch on open and once on load for the button label. Test: static markup of the button with its state and accessible name; `npm run lint && npm run build && npm test`; keyboard and both themes checked by hand.

## 3. Docs

- [ ] 3.1 Backend guide (endpoint, allowlist), frontend guide (quality dialog, manual checks), architecture overview (`GET /quality`, summary file), `evals/README.md` (`PLUMBLINE_EVAL_SUMMARY`, sampled runs don't count, Docker mount). Test: `python scripts/check_links.py` passes.
