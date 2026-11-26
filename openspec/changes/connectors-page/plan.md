## Task
8.2 `quality/useQuality.ts`: remove the unused `refresh` (and the `version` state and `useCallback` that only it needs); the hook returns `{ quality, message }` and fetches once on mount (review finding, yagni)

## Scope
- `frontend/src/quality/useQuality.ts`: delete `refresh`, `version`/`setVersion` and the `useCallback` import; the effect keeps its stale guard and runs once (`[]` dependencies); return `{ quality, message }`; update the doc comment ("fetched on mount").
- Nothing calls `refresh`: `App` passes the hook's result to `AppShell`, which passes `quality.quality` and `quality.message` to the page. Verified by search; no code or test uses `quality.refresh`.
- The two test helpers that build a `quality` prop for `AppShell` (`shell/AppShell.test.tsx`, `shell/a11y.test.tsx`) pass `refresh: () => {}` inside an object literal typed as the hook's return type. After the change that literal has an extra property, which TypeScript rejects, so both lose that one property.
- Docs: the frontend guide already says "fetched once in App and passed to the Connectors page", and design.md says the same; no doc change needed.
- Guide read: `docs/guides/frontend-development-guide.md` (1.1 tree text, 7 tests).

## Not in scope
- `qualityOutcome` and its tests (unchanged).
- `useDocuments`'s own `refresh` (used by the Documents page upload flow).
- Any behaviour change: the page already loads `/quality` once on mount.

## Files
| File | Change |
| --- | --- |
| `frontend/src/quality/useQuality.ts` | remove `refresh`, `version`, `useCallback` |
| `frontend/src/shell/AppShell.test.tsx` | drop `refresh` from the `quality` prop |
| `frontend/src/shell/a11y.test.tsx` | drop `refresh` from the `quality` prop |

## Tests
- No new test: removal of unused code. The existing `qualityOutcome` tests, the `AppShell` and a11y suites and `npm run build` (typecheck) are the evidence; a leftover `refresh` use anywhere would fail the build.

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- `grep -rn "quality.refresh\|useQuality" frontend/src` shows no `refresh` use.

## Risks / Questions
- None. The dev server hot-reloads this file, and nothing visible changes.
