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
  returns `[{name, extensions, contract_version}]` sorted by name.
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
  a note on each card when `config_matches` is false. The sidebar Quality
  item uses the overall result with the same words. Each label has an icon
  and text, never colour alone.
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

### Quality data for the sidebar

- **Choice**: One `useQuality()` hook (fetch `/quality`) used by the sidebar
  label and the page, like `useDocuments()` from #33.
- **Alternatives**: Fetch in each consumer.
- **Why**: One fetch, consistent label.
- **ADR**: none.

## Interfaces

```python
@app.get("/connectors")
def connectors(
    connectors: Annotated[dict[str, FileConnector], Depends(get_connectors)],
) -> list[dict]:
    return [
        {"name": c.name, "extensions": list(c.extensions), "contract_version": c.contract_version}
        for c in sorted(connectors.values(), key=lambda c: c.name)
    ]
```

```ts
// connectors/api.ts
export type Connector = { name: string; extensions: string[]; contract_version: number }
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
