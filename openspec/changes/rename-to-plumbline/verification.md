Verified on 2026-11-26 at commit `54650ed` (the pushed head of `feat/11-rename-to-plumbline`, with `main` merged in). Task 5.2 is a post-merge owner step by design and stays open; every other task is done.

## Checks Run

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | All checks passed |
| `cd backend && uv run ruff format --check .` | 53 files already formatted |
| `cd backend && DATABASE_URL=<local db created before the rename> uv run pytest -q -rs` | 146 passed, 1 skipped (`test_models.py:66`, needs a real Ollama model) |
| `cd backend && uv run plumbline-eval --help` | prints usage |
| `cd backend && uv build --offline` | built `plumbline_rag-0.0.0.tar.gz` and `plumbline_rag-0.0.0-py3-none-any.whl` |
| `cd frontend && npm ci && npm run lint && npm run build` | passed; `dist/` has `favicon.svg`, `favicon.ico`, `apple-touch-icon.png`; `dist/index.html` has `<title>Plumbline</title>` |
| `cd frontend && npm test` | 10 files, 30 tests passed |
| `python scripts/check_links.py` | links ok (66 files) |
| `openspec validate rename-to-plumbline` and `openspec validate --all` | valid; 3 passed, 0 failed |
| CI on the PR at `54650ed` (`gh pr checks 15`) | backend, docker, docs, frontend, openspec all pass; CI uses the new database defaults |
| Full eval, `plumbline-eval --report ../evals/reports/post-rename.json`, on the reference Apple M1 | exit 1, 3017 s; see Eval Results |
| Upgrade check on the M1 (old checkout of `main` creates a database and one document; this branch opened with `DATABASE_URL` set to the old values) | same document before and after; control run failed with `password authentication failed for user "plumbline"` |

The local backend pytest run used `DATABASE_URL` for a database created before the rename, because the new defaults do not log in to it (this is the documented upgrade behaviour). The new defaults themselves are exercised by CI.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Fresh clone quickstart | **Partial.** `test_old_names.py::test_old_names_only_in_allowed_places` guards every documented name; the upgrade run on the M1 started the app and Alembic under the new package and variable names. A fresh-clone walkthrough of the README was not run end to end. |
| Old env file | `test_config.py::test_each_old_variable_names_its_replacement` (all 7 variables), `::test_all_offenders_are_listed_and_empty_values_count`, `::test_app_refuses_to_start`, `::test_eval_cli_stops_without_a_gate_result`; `test_migrations.py::test_legacy_variable_fails_before_the_dimension_is_read` |
| Third-party connector on old group | `test_registry.py::test_old_group_connector_is_skipped_with_warning`, `::test_valid_connector_on_old_group_alone_is_not_loaded` |
| Upgrade with old database | **Manual only**, on the M1 (above); no automated test. Only the `DATABASE_URL` form of the README note was run; the Compose form (`POSTGRES_*` in `.env`) was not. |
| Conformance and eval after rename | Conformance: `test_pdf_connector.py::test_pdf_conformance`, `test_docx_connector.py::test_docx_conformance`. Eval: see Eval Results. |
| Repository search | `test_old_names.py::test_old_names_only_in_allowed_places` |
| Build | **Manual only**: `uv build` output above shows `plumbline_rag-*`; no automated test. |
| Built UI carries the icons | `test_brand_assets.py::test_index_html_icons_exist_in_public`; build output checked by hand (above) |
| README logo resolves | `test_brand_assets.py::test_readme_logos_exist` |

## Eval Results

Baseline: `evals/baselines/rag-foundation/m1-7.4-t030.summary.json` (rag-foundation task 7.4, M1, commit `476445a`). Run A: `evals/baselines/rename-to-plumbline/post-rename.summary.json` (M1, commit `54650ed`, 3017 s). Models (`llama3.1:8b` chat, `nomic-embed-text`, `qwen2.5:7b` judge), threshold 0.3, top_k 5, sample none and `golden_version` of both connectors are equal.

| Metric | Baseline -> run A | Pass rule met? |
| --- | --- | --- |
| docx hit_rate, abstention | 0.967 -> 0.967, 1.0 -> 1.0 | yes (exact) |
| pdf hit_rate, abstention | 1.0 -> 1.0, 1.0 -> 1.0 | yes (exact) |
| docx faithfulness | 0.897 -> 0.885 (-0.012) | yes |
| docx correctness | 0.767 -> 0.667 (-0.100) | **no** (outside +/-0.05) |
| pdf correctness | 0.700 -> 0.833 (+0.133) | **no** |
| pdf faithfulness | 0.875 -> 0.962 (+0.087) | **no** |

Context, not an excuse: `models.py` sets no temperature or seed, so answers vary run to run. Against the Windows run on the same unrenamed code, all 40 retrieval hits per connector are identical, only 17 (docx) and 16 (pdf) answer texts are identical, and correctness flips go both ways (docx 3 down and 3 up, pdf 1 down and 4 up). The baseline itself is below the 0.80 correctness gate on both connectors, so this change is judged against the baseline, not 0.80.

No second run was made to measure the run-to-run spread.

## Deviations

- **Waiver (product owner, 2026-11-26):** the three judge metrics above break the +/-0.05 rule and are waived as sampling noise, on the evidence above. This is a waiver, not a pass.
- Scenarios with no automated test: Build, Upgrade with old database, and the walkthrough part of Fresh clone quickstart. Evidence is manual and recorded above. These are not waived.
- The ADR 0004 and a separate migration note were dropped during implementation (product owner); the README carries a short upgrade note and the spec, design and tasks were updated to match.
- The comparison baseline was switched from the Windows CPU run to the rag-foundation M1 run (Windows judge metrics differ from the M1 by up to 0.10 on equivalent code); the Windows run is kept as `evals/baselines/rename-to-plumbline/windows-cpu.summary.json`.
- Brand assets (task 4.4) and the repo-wide old-names guard were added to scope after triage; the issue was updated.
- The fixture corpus (8 binaries) was regenerated; beyond the author string they differ in compressed bytes only, checked part by part.

## Verdict

**FAIL** under the verdict rules: three eval metrics break the pass rule, and three scenarios lack an automated test. The three eval metrics are waived by the product owner (above). The three uncovered scenarios are listed under Deviations and still need an explicit decision before merge.
