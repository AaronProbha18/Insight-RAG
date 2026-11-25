## Context

Today: `top_k=5` is a default in `retrieval.retrieve`, `graph.build_graph`
and the eval runner; the threshold comes from
`PLUMBLINE_ABSTAIN_THRESHOLD` or `graph.DEFAULT_THRESHOLD = 0.3`; the chat
model from `PLUMBLINE_LLM` (`models.chat_model()`, a `provider:model`
string for `init_chat_model`). Chunking is `chunk(doc, size=1000,
overlap=150)`, called by ingestion with the defaults. The embedding model
(`PLUMBLINE_EMBEDDINGS`) and the vector size (`PLUMBLINE_EMBEDDING_DIM`,
fixed in migration `0001`) are corpus-wide. The eval runner builds a
temporary database per connector and writes `<report>.summary.json`
(#19). Migrations `0001`, `0002` exist on this stack.

## Decisions

### Presets hold query-time settings only

- **Choice**: top-k, threshold, chat model.
- **Alternatives**: also chunking (new uploads only); also embeddings.
- **Why**: product owner: a preset must describe what is running.
  Chunking/embeddings need re-indexing, so they are index settings,
  changed through an approval step (F43), not per preset or per message
  (F06 answer modes switch presets per message).
- **ADR**: none yet - F43 will need one for index settings.

### Versions are append-only

- **Choice**: `preset_versions` rows are inserted, never updated, except
  the score columns. Saving settings equal to the latest version is a
  no-op (returns that version).
- **Alternatives**: update in place with history table.
- **Why**: AC "earlier versions remain readable and unchanged".
- **ADR**: none

### One global active preset, read per request

- **Choice**: `presets.active` with a partial unique index (at most one
  true). `/chat` loads the active preset's latest version each request
  (one indexed query) and builds the graph with its top-k, threshold and
  `init_chat_model(chat_model)`. Chat model objects cached by model
  string.
- **Alternatives**: cache the preset and reload on signal.
- **Why**: activation via CLI takes effect without restart.
- **ADR**: none

### Env seeds, preset wins

- **Choice**: on startup, if no presets exist, create `default` v1 from
  `PLUMBLINE_LLM`, the threshold env/default and top-k 5, active. If
  presets exist and `PLUMBLINE_LLM` or `PLUMBLINE_ABSTAIN_THRESHOLD`
  differs from the active version, log a warning naming each ignored var.
- **Why**: product owner choice.
- **ADR**: none

### Validation

- **Choice**: top-k integer 1..50; threshold 0 <= t <= 1; chat model in
  `PLUMBLINE_ALLOWED_LLMS` (comma-separated, default `PLUMBLINE_LLM`);
  name `^[a-z0-9][a-z0-9-]{0,39}$`. One `PresetError` with a reason per
  failed field.
- **Alternatives**: format + provider check only.
- **Why**: deterministic, no network; typos caught at save.
- **ADR**: none
- Assumption: upper bound 50 for top-k (prompt size); change if needed.

### Interface

- **Choice**: `plumbline-preset list | show NAME[@V] | save NAME
  --top-k --threshold --chat-model | activate NAME`; read-only
  `GET /presets`. No HTTP writes.
- **Why**: product owner choice; no unauthenticated way to switch the
  model.
- **ADR**: none

### Eval score per version

- **Choice**: `plumbline-eval --preset NAME[@V]` (default active) uses
  that version's top-k, threshold and chat model, then writes the
  summary's `result`, per-connector gated metrics and `started_at` to
  `preset_versions.eval_result` / `evaluated_at` in the app database
  (`DATABASE_URL`), replacing the previous score. Sampled runs and
  judge-stopped runs (exit 2) store nothing - not gate results (same rule
  as #14). The run's summary JSON gains `preset` (`name`, `version`).
- **Why**: AC "replacing its previous score"; last score only.
- **ADR**: none

### Per-document chunking record

- **Choice**: `documents.chunk_size`, `documents.chunk_overlap` (NOT
  NULL), backfilled 1000/150 for existing rows (what they were chunked
  with); ingestion writes the values it used.
- **Why**: product owner: cheap now, F43 needs it.
- **ADR**: none

## Interfaces

```python
# plumbline/presets.py
@dataclass(frozen=True)
class PresetVersion:
    name: str; version: int; top_k: int; threshold: float; chat_model: str
    eval_result: dict | None; evaluated_at: datetime | None

class PresetError(ValueError): ...           # .reasons: list[str]

def ensure_default(conn, env=os.environ) -> None
def active(conn) -> PresetVersion
def get(conn, name: str, version: int | None = None) -> PresetVersion
def save(conn, name: str, *, top_k, threshold, chat_model, env=os.environ) -> PresetVersion
def activate(conn, name: str) -> None
def list_all(conn) -> list[PresetVersion]     # every version
def record_score(conn, name, version, summary: dict) -> None
def main() -> None                            # plumbline-preset
```

```json
// GET /presets -> 200
[{"name": "default", "version": 1, "active": true,
  "settings": {"top_k": 5, "threshold": 0.3, "chat_model": "ollama:llama3.1:8b"},
  "eval": null}]                                      // null = "not evaluated"
// evaluated: "eval": {"result": "PASS", "run_at": "...", "connectors": {"pdf": {...}}}
```

## Data

Migration `0003_presets.py`:

```sql
CREATE TABLE presets (
  id         bigserial PRIMARY KEY,
  name       text UNIQUE NOT NULL,
  active     boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX presets_one_active ON presets (active) WHERE active;
CREATE TABLE preset_versions (
  id           bigserial PRIMARY KEY,
  preset_id    bigint NOT NULL REFERENCES presets(id) ON DELETE RESTRICT,
  version      int NOT NULL CHECK (version >= 1),
  top_k        int NOT NULL CHECK (top_k BETWEEN 1 AND 50),
  threshold    double precision NOT NULL CHECK (threshold BETWEEN 0 AND 1),
  chat_model   text NOT NULL,
  eval_result  jsonb,
  evaluated_at timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (preset_id, version)
);
ALTER TABLE documents
  ADD COLUMN chunk_size int NOT NULL DEFAULT 1000,
  ADD COLUMN chunk_overlap int NOT NULL DEFAULT 150;
ALTER TABLE documents ALTER COLUMN chunk_size DROP DEFAULT,
                      ALTER COLUMN chunk_overlap DROP DEFAULT;
```

## Risks / Open Questions

- Answers now depend on database state: a wrong seed silently changes
  behaviour. Mitigated by the eval-plan baseline comparison.
- The eval runner writes to the app database; it must not run against a
  production database by accident - it only updates score columns.
- #14's panel keeps showing the latest summary file, not the active
  preset's score (non-goal; follow-up).
- Depends on the stack below (#11 rename, #13 migration `0002`).
