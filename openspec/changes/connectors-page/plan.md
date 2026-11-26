## Task
6.4 Sidebar: one item "Connectors and quality" in Knowledge; remove the Quality item, the Insights group, the `quality` prop (Sidebar, SidebarDrawer, AppShell) and `quality/label.ts` with its test; `App` keeps `useQuality` for the page; update Sidebar, SidebarDrawer, AppShell and a11y tests

## Scope
- `shell/Sidebar.tsx`: the Connectors link's text becomes "Connectors and quality" (and so its collapsed tooltip and screen-reader text; same `PlugIcon`, same link to `/app/connectors`, `aria-current` on its page). Remove the Quality link and the whole Insights group, the `quality?` prop, the `qualityLabel` and `Quality` imports, the `qualityText`/`QualityIcon` code and the four icons only they used (`ChartNoAxesColumnIcon`, `CircleCheckIcon`, `CircleHelpIcon`, `CircleXIcon`). `pageLink` stays, now used twice.
- `shell/SidebarDrawer.tsx`: remove the `quality` prop and the `Quality` import.
- `shell/AppShell.tsx`: stop passing `quality.quality` to `Sidebar` and `SidebarDrawer`. The `quality` prop of `AppShell` itself stays: the Connectors page still takes it.
- `App.tsx`: unchanged (`useQuality()` still feeds the page).
- Delete `quality/label.ts` and `quality/label.test.ts`.
- Result, top to bottom: logo and collapse; New chat; Knowledge: Documents, Connectors and quality; Docs.
- Guides read: `docs/guides/frontend-development-guide.md` (1.4 real links with plain-click handling; 5 sidebar accessibility; 7.1 `Sidebar` is called as a function in `SidebarDrawer.test.tsx`, so the anchors must stay in `Sidebar`'s own element tree, which `pageLink` keeps).

## Not in scope
- Page, cards, `useQuality`, `App` (unchanged).
- Guide text for the sidebar (7.1) and the accessibility re-check wording (7.2). The frontend guide's sidebar bullet and folder tree are stale until 7.1.
- A sidebar quality indicator of any kind (decided: page only).

## Files
| File | Change |
| --- | --- |
| `frontend/src/shell/Sidebar.tsx` | merge to one item; remove Quality item, Insights group, prop, imports |
| `frontend/src/shell/SidebarDrawer.tsx` | remove the prop |
| `frontend/src/shell/AppShell.tsx` | stop passing quality to the sidebar and drawer |
| `frontend/src/quality/label.ts`, `label.test.ts` | delete |
| `frontend/src/shell/Sidebar.test.tsx` | update |
| `frontend/src/shell/a11y.test.tsx` | current-page count on the connectors page becomes 1; link-count comment |
| `frontend/src/shell/SidebarDrawer.test.tsx`, `AppShell.test.tsx` | only if they fail to compile or assert old text |

## Tests
- Order (spec "Open Connectors and quality from the sidebar"): logo, New chat, Knowledge, "Documents (3)", "Connectors and quality", then Docs; the words "Insights" and "Quality" as an item are absent (the title text "Connectors and quality" contains "quality", so I assert on `>Quality<` and `Insights`).
- One link to `/app/connectors` (spec "Quality opens its dialog": no Quality item, exactly one link goes to the page), with `aria-current="page"` only on the connectors page.
- Quality prop gone: `Sidebar` renders with no `quality` argument (the existing `render` helper drops it), and no "Quality: ..." text appears for any state.
- Collapsed: "Connectors and quality" kept as `sr-only` text; tooltip triggers drop from 6 to 5 (expand, New chat, Documents, Connectors and quality, Docs); expanded 0.
- Dialog still gone and the removed files stay removed: `existsSync` list gains `label.ts`.
- `a11y.test.tsx`: connectors page announces `aria-current` once (Documents page once, chat zero); every control named; the keyboard-reach test still has four links or more (Documents, Connectors and quality, Docs, "How it's measured").
- `SidebarDrawer.test.tsx` callbacks tests still pass (the anchor is still reachable in `Sidebar`'s tree).

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- `grep -rn "qualityLabel\|quality/label" frontend/src` returns nothing.

## Risks / Questions
- With the Insights group gone the sidebar loses its only "Insights" heading; the design keeps the heading for its Unanswered page, which isn't built (out of scope), so there is nothing to put under it.
- None otherwise.
