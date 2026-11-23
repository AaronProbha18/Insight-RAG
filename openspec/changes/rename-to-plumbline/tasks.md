## 1. Baseline

- [ ] 1.1 On the unrenamed code, run `insight-rag-eval` twice (`--report evals/reports/pre-rename-{a,b}.json`) and conformance; commit both reports. Test: reports exist and conformance passes.

## 2. Backend rename

- [ ] 2.1 `git mv` package to `backend/src/plumbline`; rename imports, distribution `plumbline-rag`, script `plumbline-eval`, FastAPI title; regenerate `uv.lock`. Test: full backend suite passes; `uv build` produces `plumbline_rag-*`.
- [ ] 2.2 Rename env vars to `PLUMBLINE_*` (models, app, migration `0001_initial.py`, tests, `.env.example`, Dockerfile). Test: existing tests pass with new names.
- [ ] 2.3 Add `plumbline/config.py` `check_legacy_env()`, called at app startup, in `plumbline-eval` `main()` and in `migrations/env.py`. Test: `test_config.py` - each `INSIGHT_RAG_X` set -> error naming `PLUMBLINE_X`; none set -> no error.
- [ ] 2.4 Registry group `plumbline.connectors` (pyproject + `registry.py`); warn per entry found under `insight_rag.connectors`, don't load it. Test: `test_registry.py` case with a fake old-group entry point -> skipped, warning names new group; conformance passes.

## 3. Infra and database defaults

- [ ] 3.1 Default DB credentials `plumbline` in `db.py`, `docker-compose.yml`, CI; Docker service/image names. Test: backend suite against the renamed DB; `docker compose config` shows new names.

## 4. Frontend, fixtures, docs

- [ ] 4.1 Frontend title `Plumbline` (`frontend/index.html`, any test asserting it). Test: frontend checks pass.
- [ ] 4.2 Fixture author "plumbline fixtures" in `evals/generate_corpus.py`; regenerate corpus. Test: connector suites and conformance pass.
- [ ] 4.3 Docs: README, `AGENTS.md`, guides, architecture overview, `evals/README.md`, `.github/` templates, `.claude/launch.json`, `openspec/specs/`; mark ADR 0004 Accepted; write `docs/migration/plumbline.md` (env-var table, connector group, keeping the old DB). Test: `test_old_names.py` - repo search for old names returns only allowed paths.

## 5. Eval and repo

- [ ] 5.1 Run `plumbline-eval --report evals/reports/post-rename.json`; compare to baseline per eval-plan; follow the migration note against a DB created on `main` and confirm its documents still list. Test: comparison recorded in `verification.md`.
- [ ] 5.2 After merge, owner renames the GitHub repo to `Plumbline` and replaces its description (`gh repo rename`, `gh repo edit --description`); update the local remote. Test: `gh repo view` shows new name and description; `git fetch` works.
