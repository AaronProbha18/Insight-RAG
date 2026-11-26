## 1. Backend

- [x] 1.1 SPA fallback: `GET /app/{path:path}` serves `index.html` when `PLUMBLINE_UI_DIR` is set; tests in `test_ui_mount.py` that `/app/documents` returns the HTML while `GET /documents` and an unknown path outside `/app/` do not

## 2. Routing and data

- [x] 2.1 `shell/route.ts`: `useRoute()` and `navigate()` over the History API, with tests for push, back (`popstate`) and unknown paths
- [x] 2.2 `documents/useDocuments.ts`: list, refresh and delete in one hook, extracted from `Documents.tsx`, with tests

## 3. Pages and shell

- [ ] 3.1 `documents/DocumentsPage.tsx`: upload area (reusing `Upload`'s input and outcome), title search, type filter, list with delete confirm; tests for search, filter, upload refresh and delete
- [ ] 3.2 `shell/Sidebar.tsx`: logo, collapse button (persisted in `localStorage`), New chat, Knowledge (Documents count, Connectors coming soon), Insights (Quality dialog), docs link, `aria-current`; tests
- [ ] 3.3 Responsive: below `md` the sidebar opens as a drawer from a menu button and closes on navigation; test
- [ ] 3.4 `App.tsx`: shell layout and page switch with `Thread` kept mounted, New chat resets the thread (confirm the runtime API first), upload bar and `Documents.tsx` removed; tests that the chat survives navigation, New chat clears it, and the bar is gone

## 4. Docs and accessibility

- [ ] 4.1 Update the frontend guide (code organisation, routing, `/app/` prefix) and the architecture overview's UI section
- [ ] 4.2 Accessibility pass: keyboard-only walk-through, focus rings, 4.5:1 contrast in both themes (extend `design-tokens.test.ts` if new colour pairs appear)
