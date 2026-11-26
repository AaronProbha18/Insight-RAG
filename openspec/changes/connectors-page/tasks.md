## 1. Backend

- [x] 1.1 `GET /connectors` in `api/app.py` over `get_connectors()`: name, extensions, contract version, sorted by name; tests for the built-ins, no extra fields, and a failed connector not listed; confirm `/quality` connector keys equal `FileConnector.name`

## 2. Data and routing

- [x] 2.1 `connectors/api.ts` (`listConnectors`) and `/connectors` in the Vite dev proxy, with tests
- [x] 2.2 `quality/useQuality.ts`: one hook for `/quality`, with tests
- [ ] 2.3 `shell/route.ts`: `connectors` page at `/app/connectors`, with tests

## 3. Page and shell

- [ ] 3.1 `connectors/ConnectorCard.tsx`: name, extensions, document count, three metric bars with the 0.80 mark, Passes / Below bar / Not measured (with eval command), not-current note, run date and models; tests for each state
- [ ] 3.2 `connectors/ConnectorsPage.tsx`: join `/connectors`, `/quality` and `/documents` by name, load and error states; wire into `App.tsx`; tests
- [ ] 3.3 Sidebar: Connectors becomes a link, Quality links to `/app/connectors` with the overall state in words; remove `Quality.tsx` and unused `QualityPanel` code; tests that the dialog is gone

## 4. Docs and accessibility

- [ ] 4.1 Update the backend guide (`GET /connectors`), frontend guide (connectors folder, routes) and the architecture overview
- [ ] 4.2 Accessibility pass: labels not by colour alone, 4.5:1 contrast in both themes (extend `design-tokens.test.ts` for new pairs), keyboard reach for every link
