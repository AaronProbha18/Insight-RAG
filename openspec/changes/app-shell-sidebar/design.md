## Context

`App.tsx` stacks `Upload` (a bar that also holds the Quality and Documents
dialog buttons) above `Thread`. FastAPI serves the built UI through
`StaticFiles(directory=..., html=True)` at `/`, which has no SPA fallback,
and `/documents` is already the JSON API (also proxied by Vite in dev). So
client paths must not reuse API paths, and production needs a fallback for
page URLs.

## Decisions

### Page URLs under `/app/`

- **Choice**: Chat at `/`, other pages at `/app/<page>` (`/app/documents`).
- **Alternatives**: hash URLs (`/#/documents`): no backend change, but
  rejected by the product owner in triage in favour of clean paths. Bare
  paths (`/documents`): these clash with the API.
- **Why**: Clean, shareable URLs, and the `/app/` prefix can never collide
  with an API route.
- **ADR**: none (local to this change). Revisit if a second client appears.

### SPA fallback as one explicit route

- **Choice**: `@app.get("/app/{path:path}")` returns `index.html` from
  `PLUMBLINE_UI_DIR`, registered before the static mount and only when that
  directory exists. Unknown API paths still return JSON 404s.
- **Alternatives**: subclass `StaticFiles` to fall back on any 404. Rejected
  because it would turn mistyped API calls into HTML 200s.
- **Why**: One route with a precise scope.
- **ADR**: none.

### Built-in router, no library

- **Choice**: a `useRoute()` hook over `window.location.pathname`,
  `history.pushState` and the `popstate` event, plus a `navigate(path)`
  helper. Sidebar links are real `<a href>` elements, so middle-click and
  copy-link work. Plain left-clicks are intercepted.
- **Alternatives**: TanStack Router or React Router. Rejected: there are two
  pages, and a dependency needs review for little gain.
- **Why**: About 30 lines. Add a library when nested routes or loaders are
  needed.
- **ADR**: none.

### Chat survives navigation

- **Choice**: The runtime stays in `App`, and `Thread` stays mounted but is
  `hidden` while another page shows.
- **Alternatives**: Unmount `Thread` and rely on the runtime holding the
  messages. UNVERIFIED: confirm by a test that asks, navigates and returns.
- **Why**: Keeping `Thread` mounted avoids the question and keeps the
  scroll position.
- **ADR**: none.

### Shared document count

- **Choice**: A `useDocuments()` hook in `documents/` owns list, refetch and
  delete. The shell calls it once and passes the data to the sidebar count
  and the Documents page. A successful upload triggers a refetch.
- **Alternatives**: Each consumer fetches on its own. Rejected because the
  count would go stale.
- **Why**: One source, so the count updates after upload and delete.
- **ADR**: none.

### Search and type filter on the client

- **Choice**: Filter the `GET /documents` result in memory: title contains
  the query (case-insensitive), and the type filter maps PDF to connector
  `pdf` and Word to connector `docx`.
- **Alternatives**: Query parameters on `GET /documents`. Rejected: that is
  an API change, and the list is small.
- **Why**: There is no API change.
- **ADR**: none.

### Collapse and responsive

- **Choice**: At `md` (768 px) and up, the sidebar is inline, and the
  collapse button toggles the icons-only state, stored in `localStorage` key
  `plumbline.sidebar.collapsed` inside try/catch. Below `md` it is hidden
  behind a menu button and opens as a base-ui `Dialog` drawer, which closes
  on navigation.
- **Alternatives**: A hand-rolled drawer. Rejected: base-ui is already
  installed and handles the focus trap and Escape.
- **Why**: There is no new code for focus management.
- **ADR**: none.

### New chat

- **Choice**: Navigate to `/` and reset the thread through the runtime.
- **Alternatives**: Remount the runtime by changing a React `key`.
- **Why**: Use the runtime's own reset if it has one. UNVERIFIED: confirm by
  reading the installed `@assistant-ui/react` types when planning that task,
  and fall back to the key remount.
- **ADR**: none.

### Quality and Connectors entries

- **Choice**: Quality is a sidebar button that opens the existing `Quality`
  dialog. Connectors is a non-link item marked "Coming soon" with
  `aria-disabled`.
- **Alternatives**: Hide Connectors until its page exists.
- **Why**: The issue lists it in the sidebar, and its page is a separate
  issue.
- **ADR**: none.

## Interfaces

```ts
// shell/route.ts
export type Page = 'chat' | 'documents'
export function useRoute(): Page            // from location.pathname; unknown -> 'chat'
export function navigate(path: string): void // pushState + notify subscribers

// documents/useDocuments.ts
export function useDocuments(): {
  documents: DocumentSummary[]
  message: string
  refresh: () => void
  remove: (d: DocumentSummary) => Promise<void>
}
```

```python
# api/app.py, registered before app.mount("/", ...)
@app.get("/app/{path:path}", include_in_schema=False)
def spa(path: str) -> FileResponse:
    return FileResponse(Path(_ui_dir) / "index.html")
```

## Data

none

## Risks / Open Questions

- Vite dev serves `index.html` for unknown paths by default, so
  `/app/documents` should work in dev without config. UNVERIFIED: confirm in
  the routing task.
- The running chat model name is out of scope. A follow-up issue adds it to
  an API.
