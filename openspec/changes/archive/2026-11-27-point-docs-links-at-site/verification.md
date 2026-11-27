## Checks Run

Run in PowerShell from `D:\PROJECTS\Plumbline\frontend` (the Bash shell's lowercase `d:\` path makes Vitest fail at suite collection, with or without this change).

- `openspec validate point-docs-links-at-site` -> valid
- `npm run lint` (oxlint) -> clean
- `npm run build` (`tsc -b && vite build`) -> built, typecheck clean (chunk-size warning only)
- `npm test` -> 39 files, 312 tests passed
- `git grep -nE "github.com|\.md['\"]" frontend/src` -> only filename fixtures (`a.md`, `returns-policy.md`); no documentation link target
- Not run: backend checks (`ruff`, `pytest`); no backend file changed. No `answer-quality-change` or `db-change` label, so no eval or drift check.

## Confirms

| Spec scenario | Covered by |
| --- | --- |
| Cards for each connector | `ConnectorsPage.test.tsx` (unchanged) |
| Page frame | `ConnectorsPage.test.tsx` (unchanged) |
| Subtitle | `ConnectorsPage.test.tsx` (unchanged) |
| Connector guide opens on the docs site | `ConnectorsPage.test.tsx` asserts `https://plumbline-docs.netlify.app/guides/connector-development-guide/`; `a11y.test.tsx` asserts both external links start with the site |
| How it's measured opens on the docs site | `ConnectorsPage.test.tsx` asserts `https://plumbline-docs.netlify.app/architecture/overview/#evaluate-plumbline-eval` |

The base URL is defined once in `frontend/src/lib/docsSite.ts` and used by `Sidebar.tsx` and `ConnectorsPage.tsx`; `REPO` is gone. `Sidebar.test.tsx` still passes unchanged.

Not checked in this session: that the live site serves these URLs. The proposal records that check (trailing slash, anchor exists) from triage.

## Deviation

none

## Verdict

PASS
