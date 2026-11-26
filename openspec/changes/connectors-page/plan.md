## Task
6.1 Types: `Connector.builtin` in `connectors/api.ts`; optional counts on `ConnectorQuality` in `quality/api.ts`; fixtures and tests follow

## Scope
- `frontend/src/connectors/api.ts`: add `builtin: boolean` to `Connector` (the backend sends it since 5.1). Update the doc comment of `listConnectors` if it names fields (it doesn't).
- `frontend/src/quality/api.ts`: add `answerable_questions?: number` and `unanswerable_questions?: number` to `ConnectorQuality`. They are optional because `/quality` omits both for summaries written before 5.2 (specs "Summary without counts").
- Fixtures follow: the `Connector` literals in `connectors/api.test.ts`, `ConnectorCard.test.tsx`, `ConnectorsPage.test.tsx` and `useConnectors.test.ts` get `builtin`. The type change is what forces this (`npm run build` typechecks tests: the repo's `tsc -b` includes them, as it did when I added `quality` to `AppShell` props), so it's the same task.
- Behaviour does not change: nothing reads `builtin` or the counts until 6.2 and 6.3.
- Guides read: `docs/guides/frontend-development-guide.md` (1.1 `api.ts` holds the wire types; 3.1 relative paths; 7.2 stubbed `fetch` tests).

## Not in scope
- Showing "Built-in" or the counts (6.2), page changes (6.3), sidebar (6.4).
- Runtime validation of the response shape (the project doesn't do it for any `api.ts`).
- Guide text (7.1).

## Files
| File | Change |
| --- | --- |
| `frontend/src/connectors/api.ts` | `builtin: boolean` |
| `frontend/src/quality/api.ts` | two optional counts |
| `frontend/src/connectors/api.test.ts` | `builtin` in the sample list |
| `frontend/src/connectors/ConnectorCard.test.tsx` | `builtin` in both `Connector` literals |
| `frontend/src/connectors/ConnectorsPage.test.tsx` | `builtin` in the sample list |
| `frontend/src/connectors/useConnectors.test.ts` | `builtin` in the sample |
| `frontend/src/quality/api.test.ts` | a case: a `/quality` body with the counts parses through unchanged, and one without them does too |

## Tests
- Existing tests keep passing with the new literal shape (they assert the list passes through unchanged, so they now prove `builtin` survives `listConnectors`).
- `quality/api.test.ts`: `getQuality` returns a measured body with both counts as given, and a body without them unchanged (no defaulting to 0, per design.md "never invent numbers").

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`

## Risks / Questions
- None. This task has no visible change; its value is that 6.2 and 6.3 compile against the right types.
