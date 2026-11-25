## Context

The old name appears in the package (`backend/src/insight_rag`), every
import, `pyproject.toml` (distribution, entry-point group, script), seven
`INSIGHT_RAG_*` env vars (`LLM`, `EMBEDDINGS`, `EMBEDDING_DIM`,
`JUDGE_LLM`, `MODEL_BASE_URL`, `ABSTAIN_THRESHOLD`, `UI_DIR`), the
migration `0001_initial.py`, default DB credentials (`db.py`, compose,
CI), Docker, the frontend title, fixture metadata, docs and the GitHub
repo.

## Decisions

### Old names fail fast, no fallback

- **Choice**: Only `PLUMBLINE_*` is read. A startup check raises if any
  `INSIGHT_RAG_*` is set, listing old -> new for each.
- **Alternatives**: read old names with a deprecation warning.
- **Why**: product owner chose a single set of names; the check satisfies
  the issue's "stop with a message naming each new variable" option.
- **ADR**: none

### Old connector group is skipped with a warning

- **Choice**: `registry.load_connectors` loads `plumbline.connectors`
  only; if `insight_rag.connectors` has entries, log one warning per entry
  naming the new group.
- **Alternatives**: load old group for one release.
- **Why**: same no-fallback policy.
- **ADR**: none

### Where the env check runs

- **Choice**: one function `check_legacy_env()` in `plumbline/config.py`,
  called at app startup and in `plumbline-eval` `main()`, and in the
  migration env before reading `PLUMBLINE_EMBEDDING_DIM`.
- **Alternatives**: check inside each `os.environ.get` call site.
- **Why**: one place, one message listing all offenders.
- **ADR**: none (local to this change)

### Rename mechanics

- **Choice**: `git mv backend/src/insight_rag backend/src/plumbline`, then
  textual replace; regenerate `uv.lock`; reinstall the venv.
- **Alternatives**: keep a `insight_rag` import shim package.
- **Why**: no external importers besides connectors, which already break
  on the group rename.
- **ADR**: none

### Eval baseline

- **Choice**: run `insight-rag-eval` on `main` before any rename and commit
  the result as the baseline; regenerate fixtures with author
  "plumbline fixtures"; re-run with `plumbline-eval`.
- **Why**: proves the rename and fixture regeneration are no-ops.

## Interfaces

```python
# plumbline/config.py
LEGACY_PREFIX = "INSIGHT_RAG_"
NEW_PREFIX = "PLUMBLINE_"

class LegacyConfigError(RuntimeError): ...

def check_legacy_env(environ: Mapping[str, str] = os.environ) -> None:
    """Raise LegacyConfigError naming PLUMBLINE_<X> for each INSIGHT_RAG_<X> set."""

# plumbline/connectors/registry.py
GROUP = "plumbline.connectors"
LEGACY_GROUP = "insight_rag.connectors"  # only used to warn
```

CLI: `plumbline-eval` (same flags as `insight-rag-eval`).

## Data

No schema migration. Default connection string becomes
`postgresql://plumbline:plumbline@localhost:5432/plumbline`. Existing
installs keep data by setting `POSTGRES_USER/PASSWORD/DB` (compose) or
`DATABASE_URL` to the old values - documented in the
README. The compose volume name is derived from the
compose project, not from these defaults - UNVERIFIED - confirm by
`docker volume ls` before and after `docker compose up` on the renamed
branch.

## Risks / Open Questions

- Renaming the GitHub repo changes the remote URL; GitHub redirects old
  URLs - UNVERIFIED for this repo's ruleset; confirm with `git fetch`
  after rename. Run as the last task, after the PR merges, by the owner.
- Regenerated PDF/DOCX bytes may differ beyond author metadata
  (timestamps); eval equality is the gate, not byte equality.
