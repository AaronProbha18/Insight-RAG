## Task
1.1 Define the docs site base URL once and use it in `shell/Sidebar.tsx` and `connectors/ConnectorsPage.tsx`. Point Connector guide at `guides/connector-development-guide/` and How it's measured at `architecture/overview/#evaluate-plumbline-eval`. Remove the now-unused `REPO` constant. Update `ConnectorsPage.test.tsx` and `shell/a11y.test.tsx` to assert the site URLs. Check: `npm test` in `frontend/` passes; `git grep -nE "github.com|\.md['\"]" frontend/src` shows no documentation target (repository links only).

## Scope
- New `frontend/src/lib/docsSite.ts` exporting `DOCS_URL = 'https://plumbline-docs.netlify.app/'` (the one definition; `lib/` is where shared non-component code lives, and importing a constant from `shell/` into `connectors/` would couple two features).
- `Sidebar.tsx`: drop its local `DOCS_URL`, import the shared one.
- `ConnectorsPage.tsx`: drop `REPO`; `MEASURED_URL = DOCS_URL + 'architecture/overview/#evaluate-plumbline-eval'`, `GUIDE_URL = DOCS_URL + 'guides/connector-development-guide/'`.
- Tests assert the site URLs.
- Guide read: `docs/guides/frontend-development-guide.md` (§1 organisation, §7 testing: static markup, no jsdom).

## Not in scope
- Docs content, the docs site, new links, env-var configurability of the base URL (a single constant is what the issue asks for).
- Updating the §1 file tree in the frontend guide (it lists feature files, not `lib/`; `lib/utils.ts` is already absent from it).

## Files
| File | Change |
| --- | --- |
| `frontend/src/lib/docsSite.ts` | New: `export const DOCS_URL` |
| `frontend/src/shell/Sidebar.tsx` | Import `DOCS_URL` from `@/lib/docsSite`, remove local const |
| `frontend/src/connectors/ConnectorsPage.tsx` | Remove `REPO`; build the two URLs from `DOCS_URL` |
| `frontend/src/connectors/ConnectorsPage.test.tsx` | Lines 68, 75: expect `https://plumbline-docs.netlify.app/architecture/overview/#evaluate-plumbline-eval` and `.../guides/connector-development-guide/` |
| `frontend/src/shell/a11y.test.tsx` | Line 186: expect `href="https://plumbline-docs.netlify.app/` instead of `github.com/` |

## Tests
- Spec "Connector guide opens on the docs site": `ConnectorsPage.test.tsx` asserts the guide href.
- Spec "How it's measured opens on the docs site": same file asserts the measured href with the anchor.
- `a11y.test.tsx` external-link test: both links still carry `rel="noreferrer"` and now start with the docs site.
- `Sidebar.test.tsx` already asserts the docs-site href; it must keep passing unchanged.

## Checks to run
- `cd frontend && npm test`
- `cd frontend && npm run lint` and `npx tsc --noEmit` (confirm exact script names in `frontend/package.json` / AGENTS.md before running)
- `git grep -nE "github.com|\.md['\"]" frontend/src` — expect no documentation target (remaining hits should be repository links or test fixtures like `returns-policy.md`)

## Risks / Questions
- The `.md` grep will still match unrelated fixtures (`'a.md'`, `returns-policy.md` in document tests); those are filenames, not doc links.
- Whether `@/lib/...` alias is used for non-shadcn code: `Sidebar.tsx` already imports `@/components/...`, so the alias works.
