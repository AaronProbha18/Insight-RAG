Issue: #34

## Why

The UI never says which file types Plumbline can read, and quality results
sit in a dialog apart from the connectors they measure. The connector
registry knows what is installed, but no endpoint exposes it.

## What Changes

- New read-only `GET /connectors`: the loaded connectors as
  `[{name, extensions, contract_version}]`, sorted by name. Connectors the
  registry skips (failed to load, wrong contract version) are not listed.
- New Connectors page at `/app/connectors`, titled "Connectors and quality",
  following the app-shell design (Design artifact "Plumbline - App Shell with
  Sidebar", board "Connectors and quality"): an intro sentence, a line
  explaining the 0.80 bar with a "How it's measured" link, a grid of cards,
  and a "Build your own" card that links to the connector guide. Each card
  shows an icon, the name, a subtitle "Built-in · .pdf · N documents", a
  status pill ("Passes", "Below bar" or "Not measured", icon plus text),
  correctness, faithfulness and abstention each as a number and a bar with
  the 0.80 mark, and a footer: last checked date, the number of answerable
  and unanswerable questions, and the chat and judge models. The data comes
  from `GET /connectors`, `/quality` and `/documents`. Layout and copy follow
  the design; colours stay on the app's existing tokens.
- `GET /connectors` also returns `builtin` (true for connectors shipped in
  this repo), and the eval runner's summary and `GET /quality` carry the
  per-connector answerable and unanswerable question counts, optional, so a
  summary written before this change still shows (without the counts).
- Rules carried over from the Quality dialog: sampled or unvalidated runs
  are "Not measured" and show the eval command; a result made with a
  different model or threshold is marked as not current.
- The sidebar has one item for the page, "Connectors and quality", in the
  Knowledge group (no longer "coming soon"). There is no separate Quality
  item, no Insights group and no quality state in the sidebar.
- The Quality dialog (`Quality.tsx`) is removed, and so is the sidebar's
  quality label.

## Non-Goals

- Installing or removing connectors from the UI.
- Running evals from the UI.
- Showing planned, not-yet-built connectors.
- Changing how evals are computed or the 0.80 gate.
- Showing the design's other pages or elements (Unanswered, Settings, chat list,
  model status) and its Planned connector cards.

## Capabilities

### New

- `connectors-page`: `GET /connectors` and the Connectors page.

### Modified

- `quality-panel`: results move from the dialog to per-connector cards.
- `app-shell`: one sidebar item, "Connectors and quality", links to the page.

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | The eval runner's summary file gains two count fields; scoring, gating, retrieval and prompts are untouched, so answers cannot change |
| Connector / ingestion | No | Reads the registry's loaded connectors; `builtin` is derived from the class's module; no contract change |
| Auth / document ACLs | No | No auth exists; the endpoints expose only listed fields |
| Public API or stream contract | Yes | New `GET /connectors` (name, extensions, contract version, `builtin`); optional question counts added to `GET /quality` |
| Persistence / migration | No | none |
| New external dependency or service | No | none |

## Impact

- `backend/src/plumbline/api/app.py`: `GET /connectors`, with tests.
- `backend/src/plumbline/evals/runner.py` (summary output) and
  `backend/src/plumbline/quality.py`: per-connector question counts.
- `frontend/vite.config.ts`: proxy `/connectors` in dev.
- `frontend/src/`: new `connectors/` (api, page, card), `shell/route.ts`
  (new page), `shell/Sidebar.tsx`, `App.tsx`; `quality/Quality.tsx`,
  `QualityPanel.tsx` and `quality/label.ts` removed.
- Docs: frontend and backend guides, architecture overview.
