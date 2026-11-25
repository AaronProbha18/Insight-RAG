## 1. Data

- [ ] 1.1 Migration `0003_presets.py`: `presets`, `preset_versions` (checks, one-active partial index), `documents.chunk_size/chunk_overlap` backfilled 1000/150. Test: `test_migrations.py` - upgrade/downgrade round trip; existing documents backfilled; two active presets rejected; top-k 0 and threshold 1.5 rejected by `CHECK`.
- [ ] 1.2 Ingestion writes the chunk size and overlap it used. Test: `test_ingestion.py` - new document row holds 1000/150.

## 2. Presets core

- [ ] 2.1 `plumbline/presets.py`: `save` (v1 for a new name, new version on change, no-op when equal), `get`, `list_all`, `activate`, validation (`PresetError` reasons; `PLUMBLINE_ALLOWED_LLMS`). Test: `test_presets.py` - v1; change -> v2 with v1 unchanged; equal save no-op; each invalid field rejected with its reason; activate switches the single active preset.
- [ ] 2.2 `ensure_default` at app startup: seed `default@1` active from env/defaults when none exist; warn per differing env var otherwise. Test: seeded values; second start doesn't reseed; warning names `PLUMBLINE_LLM` / `PLUMBLINE_ABSTAIN_THRESHOLD` when they differ.

## 3. Wiring

- [ ] 3.1 `/chat` uses the active preset's top-k, threshold and chat model (model objects cached by string); env no longer read per request. Test: `test_chat_api.py` - activating a preset with top-k 2 / threshold 0.9 changes retrieval size and abstention; `/chat` payload unchanged.
- [ ] 3.2 `plumbline-preset` CLI (`list`, `show`, `save`, `activate`) and read-only `GET /presets` (`eval: null` when not evaluated). Test: `test_preset_cli.py` and `test_presets_api.py` - listing shows name, version, settings, score or not evaluated; no write endpoint exists (POST/PUT/DELETE -> 405).
- [ ] 3.3 `plumbline-eval --preset NAME[@V]` (default active): run with that version's settings; store result, per-connector gated metrics and run time on it, replacing the previous score; sampled and exit-2 runs store nothing; summary JSON gains `preset`. Test: `test_runner.py` with fakes - score stored then replaced; sampled run leaves it null; summary names the preset.

## 4. Eval and docs

- [ ] 4.1 Run the eval-plan command on the M1; compare with the #11 post-rename baseline. Test: comparison recorded in `verification.md`.
- [ ] 4.2 Docs: `.env.example` (`PLUMBLINE_ALLOWED_LLMS`, env now seeds only), backend guide (presets, index vs query settings), architecture overview (tables, `GET /presets`), `evals/README.md` (`--preset`, scores). Test: `python scripts/check_links.py` passes.
