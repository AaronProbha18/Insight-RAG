## Checks Run

All run in this session, on branch `feat/14-quality-panel` at `bdb6afc`.

| Command | Result |
| --- | --- |
| `openspec validate quality-panel` | Change 'quality-panel' is valid |
| `cd backend && uv run ruff check .` | All checks passed |
| `cd backend && uv run ruff format --check .` | 61 files already formatted |
| `cd backend && uv run pytest -q` | 199 passed, 1 skipped (Postgres up; fresh DB after `down -v`) |
| `cd frontend && npm run lint` (oxlint) | clean, no output |
| `cd frontend && npm run build` (tsc -b + Vite) | built, no errors |
| `cd frontend && npm test` (from `D:\`, upper-case drive) | 18 files, 71 passed |
| `python scripts/check_links.py` | links ok (97 files) |

Not run: a live sampled `plumbline-eval` (task 0.1's test wording). The runner's summary file is covered by `backend/tests/test_runner.py`, which passes in the full suite; no Ollama judge run was done here.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Both connectors passed | `test_quality.py::test_both_pass`; `test_quality_api.py::test_measured_has_summary_fields_only`; `QualityPanel.test.tsx` "shows PASS, three metrics, the run date and model ids for each connector" |
| A gated metric failed | `test_quality.py::test_docx_fails_faithfulness`; `test_quality_api.py::test_docx_fails_faithfulness`; `QualityPanel.test.tsx` "names the failed metric as below 0.80" |
| No valid run | `test_quality.py::test_not_measured` (exit2, missing_field, malformed, missing file); `test_quality_api.py::test_missing_file_is_not_measured`; `QualityPanel.test.tsx` "says not measured, shows the command and no numbers" |
| Sampled run | `test_quality.py::test_not_measured[sampled]` |
| Different configuration | `test_quality.py::test_config_difference` (chat, embeddings, threshold), `test_judge_not_compared`; `test_quality_api.py::test_different_chat_model_is_flagged`; `QualityPanel.test.tsx` "warns about exactly the differing configuration" |
| Summary fields only | `test_quality.py::test_no_internal_fields_leak`; `test_quality_api.py::test_measured_has_summary_fields_only`; `/chat` unchanged: `test_chat_api.py` incl. `test_no_confidence_field_anywhere` (green in the full run) |
| Themes and keyboard | PARTIAL, see Deviations. Not colour alone: `QualityPanel.test.tsx` "shows PASS and FAIL as words next to an aria-hidden icon", `Quality.test.tsx` state labels with icon. Contrast: `design-tokens.test.ts` checks `--destructive`/`--background` and the other token pairs the panel uses. Keyboard and both themes: manual, no automated test |

## Eval Results

N/A per eval-plan (display only; retrieval, prompts and graph unchanged, so no eval metric can move).

## Deviations

- **Keyboard / theme scenario has no automated test** for opening by keyboard, focus return and visual theme rendering (static markup cannot cover them; design.md says so). It was checked by hand by the user for task 2.3 (reported in conversation, not run by this session), against `both_pass`, `docx_fails_faithfulness` and missing-file summaries. A reviewer should treat that facet as human-attested.
- `.env.example` entry for `PLUMBLINE_EVAL_SUMMARY` was added by the user; this session could not read the file, so its content is unverified here. Open: if the line is uncommented with an empty value, `os.environ.get` returns `""` and the route reads the current directory (still `not_measured`, default path ignored). Not fixed (out of the approved scope).
- "below 0.80" is hardcoded in `QualityPanel.tsx` (mirrors `evals/scoring.py` `GATE`; `/quality` does not send it), as the spec text does.
- Added `/quality` to the Vite dev proxy (`frontend/vite.config.ts`); not in the task text, approved in the 2.1 plan.
- Task 0.1 wording named `evals/runner.py`; the runner is `backend/src/plumbline/evals/runner.py` on `main`. `tasks.md` and `design.md` corrected. The branch was rebased onto `origin/main` (history rewritten; needs `git push --force-with-lease`, run by the user).
- Docker docs name no concrete mount path, because the Dockerfile does not pin one.

## Verdict

PASS. Every check passed; every scenario has a test except the keyboard/theme facet of "Themes and keyboard", which is covered by manual attestation (see Deviations) and called out for the reviewer.
