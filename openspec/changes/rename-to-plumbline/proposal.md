Issue: #6

## Why

"insight-rag" is a developer label: small-business users don't know what
"RAG" means, and the name says nothing about cited answers with measured
quality. The product is now **Plumbline**, and every name a user, operator
or connector author sees must say so.

## What Changes

- **BREAKING** Python package `insight_rag` -> `plumbline`; distribution
  `insight-rag` -> `plumbline-rag`; eval command `insight-rag-eval` ->
  `plumbline-eval`.
- **BREAKING** Env vars `INSIGHT_RAG_*` -> `PLUMBLINE_*`. Only the new names
  are read; if any old name is set, startup stops with a message naming
  each replacement (no fallback).
- **BREAKING** Connector entry-point group `insight_rag.connectors` ->
  `plumbline.connectors`. Connectors still registered under the old group
  are not loaded; the registry warns, naming the new group.
- Default Postgres user/password/database `insight_rag` -> `plumbline`
  (compose, CI, `db.py`), with a migration note for keeping an existing
  volume.
- UI title, FastAPI title, Docker service/image names, README, guides,
  architecture overview, `AGENTS.md`, `.github` templates, `.claude`
  launch config.
- Eval fixtures regenerated with author "plumbline fixtures"; metrics
  shown equal to a baseline captured before the rename.
- New ADR 0004 recording the names, the no-fallback policy, the entry-point
  rename and the distribution name.
- GitHub repository renamed and its description replaced (scope added
  during triage; the issue listed it as a non-goal).

## Non-Goals

- Changing behaviour, prompts, thresholds, chunking, or the `/chat` /
  `/documents` wire format.
- Publishing to PyPI or a container registry.
- A logo or visual identity.
- Rewriting history: `openspec/changes/**` (incl. `rag-foundation` and
  `archive/`) keeps the old names.

## Capabilities

### New

- `product-naming`: the names the product exposes, how old names are
  rejected, and how existing installs upgrade.

### Modified

- none (behaviour of existing capabilities is unchanged)

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | Yes | No logic changes, but the eval corpus is regenerated and env-var plumbing for models changes; eval must equal the pre-rename baseline. |
| Connector / ingestion | Yes | Entry-point group renamed; third-party connectors must re-register. Built-in connectors and the fixture corpus are touched. No auth/sync/rate-limit changes. |
| Auth / document ACLs | No | No access rules touched. |
| Public API or stream contract | Yes | Env vars, entry-point group, CLI and distribution names are public; `/chat` and `/documents` unchanged. |
| Persistence / migration | Yes | Default DB credentials change; no schema migration. Existing volumes need the documented override. |
| New external dependency or service | No | |

## Impact

`backend/` (package, `pyproject.toml`, `uv.lock`, migrations, tests),
`frontend/index.html`, `Dockerfile`, `docker-compose.yml`, `.env.example`,
`.github/`, `.claude/launch.json`, `evals/`, `docs/`, `README.md`,
`AGENTS.md`, `openspec/specs/`, GitHub repo settings.
