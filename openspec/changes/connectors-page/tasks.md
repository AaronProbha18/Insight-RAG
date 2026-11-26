## 1. Backend

- [x] 1.1 `GET /connectors` in `api/app.py` over `get_connectors()`: name, extensions, contract version, sorted by name; tests for the built-ins, no extra fields, and a failed connector not listed; confirm `/quality` connector keys equal `FileConnector.name`

## 2. Data and routing

- [x] 2.1 `connectors/api.ts` (`listConnectors`) and `/connectors` in the Vite dev proxy, with tests
- [x] 2.2 `quality/useQuality.ts`: one hook for `/quality`, with tests
- [x] 2.3 `shell/route.ts`: `connectors` page at `/app/connectors`, with tests

## 3. Page and shell

- [x] 3.1 (layout superseded by 6.2) `connectors/ConnectorCard.tsx`: name, extensions, document count, three metric bars with the 0.80 mark, Passes / Below bar / Not measured (with eval command), not-current note, run date and models; tests for each state
- [x] 3.2 (layout superseded by 6.3) `connectors/ConnectorsPage.tsx`: join `/connectors`, `/quality` and `/documents` by name, load and error states; wire into `App.tsx`; tests
- [x] 3.3 (Quality link and label superseded by 6.4) Sidebar: Connectors becomes a link, Quality links to `/app/connectors` with the overall state in words; remove `Quality.tsx` and unused `QualityPanel` code; tests that the dialog is gone

## 4. Docs and accessibility

- [x] 4.1 (sidebar, page and `/connectors` text redone in 7.1) Update the backend guide (`GET /connectors`), frontend guide (connectors folder, routes) and the architecture overview
- [x] 4.2 (re-checked in 7.2) Accessibility pass: labels not by colour alone, 4.5:1 contrast in both themes (extend `design-tokens.test.ts` for new pairs), keyboard reach for every link

## 5. Backend additions (scope change after 4.2)

- [x] 5.1 `builtin` on `GET /connectors` (true when the connector class module is under `plumbline.connectors.builtin`); update `test_connectors_api.py`: built-ins true, a third-party fake false, still exactly four fields
- [x] 5.2 Runner summary writes `answerable_questions` and `unanswerable_questions` per connector; `quality.load_quality` passes them through as optional non-negative integers (absent or invalid: left out, result otherwise unchanged); tests in `test_runner.py`, `test_quality.py`, `test_quality_api.py`, including a summary without the counts

## 6. Page, cards and sidebar to the design

- [x] 6.1 Types: `Connector.builtin` in `connectors/api.ts`; optional counts on `ConnectorQuality` in `quality/api.ts`; fixtures and tests follow
- [x] 6.2 `connectors/ConnectorCard.tsx` redesign: icon, name, subtitle "Built-in · .pdf · N documents", status pill (Passes / Below bar / Not measured, icon plus text), metric rows, footer with date, question counts when present, chat and judge models; keep Not measured with the eval command and the not-current note; existing tokens only; tests for each state
- [x] 6.3 `connectors/ConnectorsPage.tsx` redesign: title "Connectors and quality", intro, 0.80 explainer with a "How it's measured" link, card grid, "Build your own" card linking to the connector guide, no Planned cards; links are `main` on `AaronProbha18/Plumbline`; keep load, error and empty states; tests
- [x] 6.4 Sidebar: one item "Connectors and quality" in Knowledge; remove the Quality item, the Insights group, the `quality` prop (Sidebar, SidebarDrawer, AppShell) and `quality/label.ts` with its test; `App` keeps `useQuality` for the page; update Sidebar, SidebarDrawer, AppShell and a11y tests

## 7. Docs and accessibility again

- [x] 7.1 Update the backend guide (`/connectors` has four fields; `/quality` counts), frontend guide (folder tree, sidebar and page accessibility text, checklist) and the architecture overview; remove the Quality-link wording added in 4.1
- [x] 7.2 Accessibility re-check: one current-page link on the Connectors page (a11y test), status pills are words plus icon, new link text is named, 4.5:1 test still passes with no new token; by-hand light/dark and keyboard check noted in `verification.md`
