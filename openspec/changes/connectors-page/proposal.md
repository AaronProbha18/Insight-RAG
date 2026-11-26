Issue: #34

## Why

The UI never says which file types Plumbline can read, and quality results
sit in a dialog apart from the connectors they measure. The connector
registry knows what is installed, but no endpoint exposes it.

## What Changes

- New read-only `GET /connectors`: the loaded connectors as
  `[{name, extensions, contract_version}]`, sorted by name. Connectors the
  registry skips (failed to load, wrong contract version) are not listed.
- New Connectors page at `/app/connectors`: one card per connector with its
  name, extensions, indexed document count, latest quality result
  (correctness, faithfulness, abstention as a number and a bar with the 0.80
  mark, labelled "Passes", "Below bar" or "Not measured"), and when and with
  which models it was last checked. The data comes from `GET /connectors`,
  `/quality` and `/documents`.
- Rules carried over from the Quality dialog: sampled or unvalidated runs
  are "Not measured" and show the eval command; a result made with a
  different model or threshold is marked as not current.
- The sidebar's Connectors item becomes a link to the page (no longer
  "coming soon"). The Quality item links to the same page and shows the
  overall state in words ("Passes", "Below bar", "Not measured").
- The Quality dialog (`Quality.tsx`) is removed; `QualityPanel` pieces are
  reused or removed as the cards need.

## Non-Goals

- Installing or removing connectors from the UI.
- Running evals from the UI.
- Showing planned, not-yet-built connectors.
- Changing `/quality` or how evals are computed.

## Capabilities

### New

- `connectors-page`: `GET /connectors` and the Connectors page.

### Modified

- `quality-panel`: results move from the dialog to per-connector cards.
- `app-shell`: Connectors and Quality in the sidebar link to the page.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Display only; `/chat` and the eval runner are untouched |
| Connector / ingestion | No | Reads the registry's loaded connectors; no contract change |
| Auth / document ACLs | No | No auth exists; the endpoint exposes only name, extensions and contract version |
| Public API or stream contract | Yes | New `GET /connectors` |
| Persistence / migration | No | none |
| New external dependency or service | No | none |

## Impact

- `backend/src/plumbline/api/app.py`: `GET /connectors`, with tests.
- `frontend/vite.config.ts`: proxy `/connectors` in dev.
- `frontend/src/`: new `connectors/` (api, page, card), `shell/route.ts`
  (new page), `shell/Sidebar.tsx`, `App.tsx`; `quality/Quality.tsx` removed.
- Docs: frontend and backend guides, architecture overview.
