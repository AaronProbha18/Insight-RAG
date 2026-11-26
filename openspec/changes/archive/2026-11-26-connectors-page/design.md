## Context

`load_connectors()` returns `dict[str, FileConnector]` and already skips,
with a logged warning, connectors that fail to load or declare a different
`contract_version`. `FileConnector` carries `name`, `extensions` and
`contract_version` as class variables. `GET /quality` returns a per-connector
result keyed by connector name, a `not_measured` state, and
`config_matches`/`config_differences`. Documents carry a `connector` field.
#33 added the sidebar, `shell/route.ts` (pages under `/app/`) and the SPA
fallback for `/app/{path}`.

## Decisions

### `GET /connectors` reuses the registry dependency

- **Choice**: A route depending on the existing `get_connectors()` that
  returns `[{name, extensions, contract_version, builtin}]` sorted by name.
- **Alternatives**: A dict keyed by name. Rejected: a list is easier to
  extend and render in order.
- **Why**: No new logic; skip rules stay in the registry.
- **ADR**: none (local to this change).

### Join on the client

- **Choice**: The page fetches `/connectors`, `/quality` and `/documents`
  and joins them by connector name. A connector absent from the quality
  summary shows "Not measured".
- **Alternatives**: A backend endpoint that merges all three. Rejected: it
  duplicates `/quality` and `/documents`.
- **Why**: Three small reads; each endpoint keeps one job.
- **ADR**: none.

### Labels

- **Choice**: Per card: "Passes" when the connector's result is PASS,
  "Below bar" when FAIL (naming each metric under 0.80), "Not measured" when
  the summary is `not_measured` or has no entry for it. Not-current shows as
  a note on each card when `config_matches` is false. Each label has an icon
  and text, never colour alone. The label is a status pill on each card.
- **Alternatives**: Keep PASS/FAIL wording. Rejected: the issue asks for
  these words.
- **Why**: Matches the issue's acceptance criteria.
- **ADR**: none.

### Metric bars

- **Choice**: A plain `<div>` bar per metric, its width set to the value,
  with a 0.80 marker line and the number as text beside it. The bar is
  `aria-hidden`; the text carries the value.
- **Alternatives**: `<meter>`. Rejected: its styling is inconsistent across
  browsers.
- **Why**: No library, readable by screen readers through the text.
- **ADR**: none.

### Quality data for the page

- **Choice**: One `useQuality()` hook (fetch `/quality`), called in `App` and
  passed to the page, like `useDocuments()` from #33.
- **Alternatives**: Fetch inside the page. The sidebar no longer shows a
  quality state (below), so only the page needs it; the hook stays where it
  is to keep the page's data flow unchanged.
- **Why**: Already built and tested; no rework for no gain.
- **ADR**: none.

### One sidebar item

- **Choice**: One item, "Connectors and quality", in the Knowledge group,
  linking to `/app/connectors`. No Quality item, no Insights group, no
  state label or pill in the sidebar. The page title is the same words.
- **Alternatives**: Keep a Quality item that links to the same page, with
  the overall state (built first in 3.3); a pill on the merged item.
  Rejected by the product owner: two links to one page, and the state is on
  the page.
- **Why**: One destination, one entry. `quality/label.ts` and the sidebar's
  `quality` prop are removed.
- **ADR**: none.

### Page layout follows the design

- **Choice**: Layout and copy follow the Design artifact "Plumbline - App
  Shell with Sidebar" (board "Connectors and quality"): header with intro,
  an explainer line with the 0.80 mark and a "How it's measured" link
  (`evals/README.md` on GitHub, `main`), a responsive card grid, and a "Build
  your own" card linking to the connector guide. The design's raw hex colours
  are not used: the page uses the existing tokens (no new colour, 4.5:1 in
  both themes). The design's "Coming next" Planned cards are not built (issue
  non-goal). Links use the current repository, not the design's old name.
- **Alternatives**: Copy the design's colours as new tokens. Rejected: the
  repo's styling rules; the product owner chose tokens.
- **Why**: Layout and copy come from the design without a new look.
- **ADR**: none.

### `builtin` on `GET /connectors`

- **Choice**: `builtin` is true when the connector class's module is under
  `plumbline.connectors.builtin` (the built-ins `pdf` and `docx` are
  registered there in `backend/pyproject.toml`; verified). The card shows
  "Built-in" only when true.
- **Alternatives**: Record the entry point's distribution in the registry
  (a registry change); a hard-coded name list. Rejected: more code or a list
  that drifts.
- **Why**: No registry change, no contract change.
- **ADR**: none.

### Question counts, optional end to end

- **Choice**: The eval runner's summary writes, per connector,
  `answerable_questions` and `unanswerable_questions` (counted from the
  scored questions, each of which keeps `answerable`: verified in
  `evals/scoring.py`). `quality.load_quality` passes them through as
  optional integers per connector, and leaves them out when absent or not
  non-negative integers (never `not_measured` for that reason). The card's
  footer says "on A answerable and U unanswerable questions" only when both
  are present.
- **Alternatives**: Show the existing total `questions`. Rejected: the
  design's wording needs the split. Backfilling old summaries. Rejected:
  numbers must not be invented.
- **Why**: Summaries written before this change keep working.
- **ADR**: none.

## Interfaces

```python
@app.get("/connectors")
def connectors(
    connectors: Annotated[dict[str, FileConnector], Depends(get_connectors)],
) -> list[dict]:
    return [
        {
            "name": c.name,
            "extensions": list(c.extensions),
            "contract_version": c.contract_version,
            "builtin": type(c).__module__.startswith("plumbline.connectors.builtin."),
        }
        for c in sorted(connectors.values(), key=lambda c: c.name)
    ]
```

```ts
// connectors/api.ts
export type Connector = { name: string; extensions: string[]; contract_version: number; builtin: boolean }

// quality/api.ts: ConnectorQuality gains optional `answerable_questions` and `unanswerable_questions` (number)
export async function listConnectors(): Promise<Connector[]>

// shell/route.ts
export type Page = 'chat' | 'documents' | 'connectors'   // /app/connectors
```

## Data

none

## Risks / Open Questions

- Registry key vs `name`: the card join assumes the `/quality` connector
  keys equal `FileConnector.name` (`pdf`, `docx`). UNVERIFIED: confirm in
  task 1.1 against the eval runner's summary.
- The runner field path is confirmed by reading (`r["questions"]` entries
  hold `answerable`), but not run end to end: task 5.2's tests confirm the
  summary shape; a real eval run (not required here) would confirm it on real
  output.
- The design artifact's links point at an old repository name; the page uses
  `AaronProbha18/Plumbline`, `main`. UNVERIFIED that `evals/README.md` and
  `docs/guides/connector-development-guide.md` keep those paths: the link
  checker covers in-repo links, not these absolute URLs.
