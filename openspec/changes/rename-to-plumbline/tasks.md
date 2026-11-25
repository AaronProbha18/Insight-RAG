## 1. Baseline

- [x] 1.1 On the unrenamed code, run `insight-rag-eval` once (`--report evals/reports/pre-rename-a.json`; judge `ollama:qwen2.5:7b`, chat `ollama:llama3.1:8b`, embeddings `ollama:nomic-embed-text`) and conformance; commit the report. Test: report exists and conformance passes.

## 2. Backend rename

- [x] 2.1 `git mv` package to `backend/src/plumbline`; rename imports, distribution `plumbline-rag`, script `plumbline-eval`, FastAPI title; regenerate `uv.lock`. Test: full backend suite passes; `uv build` produces `plumbline_rag-*`.
- [x] 2.2 Rename env vars to `PLUMBLINE_*` (models, app, migration `0001_initial.py`, tests, `.env.example`, Dockerfile). Test: existing tests pass with new names.
- [x] 2.3 Add `plumbline/config.py` `check_legacy_env()`, called at app startup, in `plumbline-eval` `main()` and in `migrations/env.py`. Test: `test_config.py` - each `INSIGHT_RAG_X` set -> error naming `PLUMBLINE_X`; none set -> no error.
- [x] 2.4 Registry group `plumbline.connectors` (pyproject + `registry.py`); warn per entry found under `insight_rag.connectors`, don't load it. Test: `test_registry.py` case with a fake old-group entry point -> skipped, warning names new group; conformance passes.

## 3. Infra and database defaults

- [x] 3.1 Default DB credentials `plumbline` in `db.py`, `docker-compose.yml`, CI; Docker service/image names; temp DB prefixes `insight_rag_test_` / `insight_rag_eval_` (`conftest.py`, `test_database.py`, `runner.py`, `test_runner.py`). Test: backend suite against the renamed DB; `docker compose config` shows new names.

## 4. Frontend, fixtures, docs

- [x] 4.1 Frontend title `Plumbline` (`frontend/index.html`, any test asserting it). Test: frontend checks pass.
- [x] 4.2 Fixture author "plumbline fixtures" in `evals/generate_corpus.py`; regenerate corpus. Test: connector suites and conformance pass.
- [x] 4.3 Docs: README, `AGENTS.md`, guides, architecture overview, `evals/README.md`, `.github/` templates, `.claude/launch.json`, `openspec/specs/`; mark ADR 0004 Accepted; write `docs/migration/plumbline.md` (env-var table, connector group, keeping the old DB). Test: `test_old_names.py` - repo search for old names returns only allowed paths.
- [x] 4.4 Brand assets: move `plumbline-brand/` to `docs/brand/`; add `frontend/public/` with `favicon.svg` (from `plumbline-icon-small.svg`), `favicon.ico`, `apple-touch-icon.png`; add the three icon links to `frontend/index.html`; add the README `<picture>` logo header. Test: a check that every icon link in `index.html` and every logo link in the README resolves to an existing file; `npm run build` output contains the three icons.
- [x] 4.5 Drop ADR 0004 and `docs/migration/plumbline.md` (product owner: not required); remove the pointers to them; add a short README note on keeping an existing database. Test: `test_old_names.py` and `check_links.py` pass.
- [x] 4.6 Merge `main` (progress bar and summary JSON in the runner, rag-foundation M1 baseline) into the branch; switch the comparison baseline to the M1 summary. Test: full suite, `test_old_names.py`, `check_links.py` and `openspec validate --all` pass after the merge.
- [x] 4.7 Keep the Windows baseline as a summary under `evals/baselines/rename-to-plumbline/` (main's convention: committed evidence is summaries only); drop the full per-question report and the "baselines are force-added" note. Test: `test_old_names.py` and `check_links.py` pass.

## 5. Eval and repo

- [x] 5.1 Run `plumbline-eval` on the M1 (run A, `evals/baselines/rename-to-plumbline/post-rename.summary.json`); compare to `evals/baselines/rag-foundation/m1-7.4-t030.summary.json` per eval-plan; follow the README upgrade note against a DB created on `main`. Test: comparison recorded in `verification.md` (three judge metrics outside the rule, waived by the product owner).
- [ ] 5.2 After merge, owner renames the GitHub repo to `Plumbline` and replaces its description (`gh repo rename`, `gh repo edit --description`); update the local remote; upload `docs/brand/social-preview.png` in Settings > General > Social preview (UNVERIFIED that no API exists; confirm). Test: `gh repo view` shows new name and description; `git fetch` works; the social preview is visible on the repo page.
