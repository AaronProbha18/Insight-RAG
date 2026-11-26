## Checks Run

All run in this session, on branch `feat/33-app-shell-sidebar` at `290f230`.

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | All checks passed |
| `cd backend && uv run ruff format --check .` | 61 files already formatted |
| `cd backend && uv run python -m pytest -q -rs` | 202 passed, 1 skipped (`tests/test_models.py:66`, needs Ollama on localhost:11434). `uv run pytest` itself fails here with "uv trampoline failed to canonicalize script path", so `python -m pytest` was used |
| `cd frontend && npm run lint` | oxlint: no warnings or errors |
| `cd frontend && npm run build` | `tsc -b` and Vite build succeeded (Vite's chunk-size notice only) |
| `cd frontend && npm test` (from `D:\`) | 24 files, 128 tests passed |
| `openspec validate app-shell-sidebar` | Change 'app-shell-sidebar' is valid |
| `openspec validate --all` | 12 passed, 0 failed |
| Vite dev, `GET /app/documents` (run during task 3.4) | HTTP 200 with the app's `index.html`, no config |

Not run: `scripts/smoke.sh` and `docker compose up --build` (no stack started for verification), and the datamodel and drift checks (no `db-change` label, no Alembic revision, no `datamodel/` change in the diff).

## Scenario Coverage

"Manual" means the check is a row of `a11y-walkthrough.md`, reported PASS by the owner. It was **not run by this verification**.

| Spec scenario | Test |
| --- | --- |
| app-shell: Open Documents from the sidebar | `shell/Sidebar.test.tsx` (Documents is `/app/documents` with `aria-current` only on that page); `shell/SidebarDrawer.test.tsx` (plain click calls `pushState`); `shell/route.test.ts` (push and notify); `shell/AppShell.test.tsx` (Documents page shown) |
| app-shell: Quality opens its dialog | `shell/Sidebar.test.tsx` (Quality control present), `quality/Quality.test.tsx` (trigger, item and compact forms). **Opening the dialog is manual only** (walk-through row 8) |
| app-shell: Back returns to the chat | `shell/route.test.ts` ("popstate (back) notifies and returns to chat"); manual row 3 |
| app-shell: Shared link in production | `backend/tests/test_ui_mount.py::test_app_path_serves_index`; `shell/route.test.ts` (`pageFor('/app/documents')`); Vite dev 200 |
| app-shell: API paths are not affected | `backend/tests/test_ui_mount.py::test_api_and_unknown_paths_are_not_index` and `::test_no_app_route_without_ui_dir` |
| app-shell: Return to an unchanged chat | `shell/AppShell.test.tsx` (chat stays in the markup, inside a `hidden` element, on the Documents page). **Behaviour is manual only** (row 3) |
| app-shell: New chat clears it | **No automated test.** Needs a DOM and a live runtime (accepted at plan time for 3.4, no jsdom). Manual row 11 only |
| app-shell: Narrow window | `shell/AppShell.test.tsx` (`hidden md:block` wrapper, "Open menu"), `shell/SidebarDrawer.test.tsx` (menu button `md:hidden`, closed drawer, close on navigation). Drawer behaviour below 768 px is manual (row 10) |
| app-shell: Collapse persists | `shell/collapsed.test.ts` (round-trip, storage errors), `shell/Sidebar.test.tsx` (collapsed markup). The reload itself is manual (row 9) |
| app-shell: Keyboard only | `shell/a11y.test.tsx` (accessible names, one `aria-current`, no outline removal, no colour literals), `design-tokens.test.ts` (contrast pairs, both themes); keyboard walk-through rows 1-12 manual |
| chat-ui: Upload outcome is announced | `documents/DocumentsPage.test.tsx` (labelled file input, `role="status"`), `uploadAndRefresh` tests (outcome returned). Announcement itself manual (row 7) |
| chat-ui: Upload bar is gone | `App.test.tsx` ("no upload bar above the chat"), `shell/AppShell.test.tsx` ("no upload bar on any page but Documents") |
| document-management: Two documents | `documents/DocumentsPage.test.tsx` ("lists documents newest first with type and delete buttons") |
| document-management: No documents | `documents/DocumentsPage.test.tsx` ("shows the empty state that points to upload") |
| document-management: Search and filter | `documents/DocumentsPage.test.tsx` (`filterDocuments`: search any case, filter by type, combined) |
| document-management: Count follows upload and delete | `uploadAndRefresh` (refresh only for `indexed`), `useDocuments.test.ts` (`deleteOutcome`: 204, 404, failure, encoded id), `Sidebar.test.tsx` (count shown). The hook's refetch effect is not unit-tested; manual rows 6-7 |

**Gap:** "New chat clears it" has no automated test. Three more scenarios (Quality dialog opening, Return to an unchanged chat, Count follows upload and delete) have automated structural tests but rest on the manual walk-through for the behaviour itself.

## Eval Results

N/A per eval-plan (UI layout and a static-file route; nothing on the answer path changes).

## Deviations

- **Added after planning, at the owner's request, not in the specs or `design.md`:** the logo-to-expand-icon swap on hover or keyboard focus when collapsed, tooltips on collapsed items, Quality restyled as a sidebar row (`itemClassName` on `Quality`), more spacing between collapsed icons. The specs and `design.md` do not mention them.
- **New Chat** uses the `design.md` fallback (React `key` remount) because `runtime.thread.reset()` could not be verified in Node. `design.md` named this fallback.
- **Task 3.1:** the two pure functions are in `documents/pageLogic.ts` (extra file) to satisfy the `only-export-components` lint rule.
- **Task 3.3:** the drawer's callback tests live in `SidebarDrawer.test.tsx`, not `Sidebar.test.tsx`.
- **Docs link** points to the GitHub URL of `docs/architecture/overview.md` (default taken at plan time; owner approved). The logo is `public/favicon.svg`.
- **Open question for the owner:** `--ring` against `--background` is about 2.6:1 (light) and 4.2:1 (dark) at full opacity, below WCAG's 3:1 guideline for non-text elements in light mode. The spec asks only for a visible ring and 4.5:1 text. Unchanged; the value was computed by hand, not by a test.
- **Frontend guide** §1.1 tree still omits `feedback/` (predates this change, left alone).

## Verdict

PASS on every automated check. One caveat the reviewer must decide: **"New chat clears it" has no automated test**, and under this skill's strict rule (a scenario without a test is a FAIL) that is a gap. It was an accepted limit at plan time (no jsdom), and the owner reported manual row 11 as PASS, but this verification did not run it. If the owner does not accept manual-only evidence for that scenario, the fix is to add jsdom with testing-library (new dependency, needs approval) and a test that asks, presses New chat and sees an empty thread.
