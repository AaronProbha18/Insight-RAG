Issue: #14

## Why

The project promises measured answer quality, but the eval result lives
in a gitignored JSON file, so people using the app can't see whether
their setup passed, or by how much.

## What Changes

- New read-only `GET /quality`: reads the eval summary file written by the
  runner (`<report>.summary.json`, PR #19) from `PLUMBLINE_EVAL_SUMMARY`
  (default `evals/reports/latest.summary.json`) and returns an allowlist
  of summary fields: overall state, run date, model ids, threshold, and
  per connector PASS/FAIL, failed metric names and the three gated
  metrics. Also returns whether the evaluated configuration matches the
  running one.
- "Not measured yet" when the file is missing or unreadable, the run
  stopped at the judge gate (exit 2), or it was a sampled run
  (`--sample`).
- Chat UI: a "Quality: PASS / FAIL / Not measured" button in the upload
  bar opens a dialog with the panel; a warning when the result is for a
  different configuration; the eval command when not measured.
- Docs: backend guide, frontend guide, architecture overview,
  `evals/README.md`.

## Non-Goals

- Running evals from the UI.
- A history of eval runs.
- Golden sets per document library (Health Score).
- Per-answer confidence; `/chat` is unchanged.

## Capabilities

### New

- `quality-panel`: exposing the latest eval result and showing it in the
  chat UI.

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Displays an existing eval result; nothing in answering changes. |
| Connector / ingestion | No | |
| Auth / document ACLs | No | Summary is non-sensitive and allowlisted; no per-question text. |
| Public API or stream contract | Yes | New `GET /quality`; `/chat` unchanged. |
| Persistence / migration | No | Reads a file; no database change. |
| New external dependency or service | No | Uses the installed Base UI Dialog. |

## Impact

`backend/src/plumbline/api/app.py`, new `backend/src/plumbline/quality.py`,
backend tests and a fixture summary; `frontend/src/` (new `quality/`
folder, upload bar), frontend tests; `.env.example`, Docker docs;
backend/frontend guides, architecture overview, `evals/README.md`.
Depends on PR #19 being in the branch (see tasks 0.1).
