Issue: #23

## Why

Answer settings live in env vars and code defaults, so trying another
combination means editing and restarting, and nothing records which
combination earned which eval score. Feature ref: F42, Recipes v0.

## What Changes

- Named, versioned **presets** in the database. A preset version holds
  the query-time settings only: **top-k, abstain threshold, chat model**.
  Saving changed settings creates a new version; earlier versions are
  never modified.
- One global **active preset**; `/chat` reads its latest version on every
  request for top-k, threshold and chat model.
- On first start with no presets, a `default` preset (version 1) is
  seeded from today's values (`PLUMBLINE_LLM`,
  `PLUMBLINE_ABSTAIN_THRESHOLD` or `DEFAULT_THRESHOLD`, top-k 5) and made
  active. After that the preset wins; startup warns for each env var that
  differs and is ignored.
- Validation on save: top-k >= 1 (upper bound in design), threshold in
  0..1, chat model in the `PLUMBLINE_ALLOWED_LLMS` allowlist (default:
  the current `PLUMBLINE_LLM`). Rejections carry a reason.
- `plumbline-preset list|show|save|activate` CLI; read-only
  `GET /presets`.
- `plumbline-eval --preset name[@version]` (default: active) evaluates a
  preset version and stores its last score and run time on it, replacing
  the previous one. Unevaluated versions list as "not evaluated".
- Each document records the chunk size and overlap it was indexed with
  (needed by F43 re-index).

## Non-Goals

- Chunking and the embedding model in presets: they are **index
  settings** (changing them needs re-indexing) and stay in env/code until
  F43. Scope cut from the issue during triage.
- UI for presets; any HTTP endpoint that changes presets.
- Per-user, per-tenant or per-message presets (answer modes, F06, later).
- Automatic tuning; score history (last score only).
- Showing the active preset's score in the #14 quality panel.
- Removing env vars for secrets or connection settings.

## Capabilities

### New

- `answer-presets`: versioned query-time settings, activation, and their
  last eval score.

### Modified

- none (answers use the active preset's values; behaviour with the
  seeded default is unchanged)

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | Yes | top-k, threshold and chat model now come from the database; a seeding or wiring mistake changes answers. Eval with the default preset must equal the env-configured baseline. |
| Connector / ingestion | Yes | Ingestion records chunk size/overlap per document; chunking itself unchanged. |
| Auth / document ACLs | No | No identity. Changing presets needs shell access (CLI); HTTP is read-only. |
| Public API or stream contract | Yes | New `GET /presets`, new CLI, new `plumbline-eval --preset`; `/chat` payloads unchanged. |
| Persistence / migration | Yes | Migration `0003`: `presets`, `preset_versions`, document chunking columns (backfilled with today's defaults). |
| New external dependency or service | No | |

## Impact

`backend/src/plumbline/` (new `presets.py`, `api/app.py`, `ingestion.py`,
`evals/runner.py`, `pyproject.toml` script), migration `0003`, backend
tests, `.env.example`, backend guide, architecture overview,
`evals/README.md`.
