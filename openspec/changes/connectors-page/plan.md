## Task
7.2 Accessibility re-check: one current-page link on the Connectors page (a11y test), status pills are words plus icon, new link text is named, 4.5:1 test still passes with no new token; by-hand light/dark and keyboard check noted in `verification.md`

## Scope
A re-audit of what changed since 4.2 (card redesign, page frame, external links, sidebar merge), with guards for what the existing tests do not reach. Findings from reading the code:

- **Already guarded, still true:** one `aria-current` on the connectors page (6.4's a11y change); every control named (`unnamed()` over `AppShell` markup); no `outline-none` without `focus-visible:` and no colour literals in the shell and connectors sources (both files are in the scan list).
- **Contrast:** the card and page use only `text-foreground`, `text-muted-foreground`, `text-destructive`, `border-border`, `border-destructive`, `bg-foreground`, `bg-muted` and `bg-background`. Every text/background pair they form is already in `PAIRS` (`--foreground`/`--background`, `--muted-foreground`/`--background`, `--destructive`/`--background`, `--foreground`/`--muted`). `--destructive` on `--muted` is **not** used (6.2 dropped it after it measured 4.37:1). So no new token and no new pair.
- **Gap 1:** `AppShell`'s static render of the connectors page only shows the loading state, so the cards, the status pills and "Build your own" are never exercised by the a11y suite.
- **Gap 2:** nothing stops a later change from adding a colour class whose pair is not measured.

Changes, all tests:
1. `shell/a11y.test.tsx`: render `ConnectorsView` loaded, with a passing, a failing and a not-measured connector, and check: every link and control has a name (reuse `unnamed()`); each card is a `<section>` with an `aria-label`; the headings run h1 then h2 with no skipped level; each status pill has an `aria-hidden` icon next to its word (Passes, Below bar, Not measured); the two external links have `href`, `rel="noreferrer"` and visible text; no element relies on `tabindex` or `aria-disabled`.
2. `shell/a11y.test.tsx`: a source guard for the page links: the `link` class string in `ConnectorsPage.tsx` contains `focus-visible:`; and a guard that the colour utilities in `ConnectorCard.tsx` and `ConnectorsPage.tsx` are only from the allowed set above, so a new colour forces a `PAIRS` update. (Spacing and size classes like `text-sm` are not colours; the check matches only `text-`, `bg-` and `border-` followed by a token name.)
3. `design-tokens.test.ts`: no change (it must still pass as is; I'll say so in the report).
4. By-hand check (not runnable from here): in the running app, `/app/connectors` in light and dark OS themes, Tab through the sidebar, Enter on "Connectors and quality", Tab reaches "How it's measured" and "Connector guide" with a visible ring, and the three card states read correctly. I will not mark this done for you: either you do it and tell me, or `verification.md` (written by `rag-verify`) records "not checked by hand".

Guides read: `docs/guides/frontend-development-guide.md` §5 accessibility, §6 styling (pairs and tokens), §7.1 (what is checked by hand).

## Not in scope
- Any visual or markup change (this task adds guards, not fixes; if a guard fails on real markup, I stop and report it rather than change the page silently).
- Screen-reader testing with real assistive tech.
- `verification.md` (the next step, `rag-verify`).

## Files
| File | Change |
| --- | --- |
| `frontend/src/shell/a11y.test.tsx` | loaded-page checks; link-focus and colour-class source guards |

## Tests
These are the tests (spec "Themes and keyboard": Passes and Below bar not by colour alone, text meets 4.5:1, every link reachable). Existing `design-tokens.test.ts` is the contrast evidence.

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- `python scripts/check_links.py`

## Risks / Questions
- The colour-class guard is a small regex over two source files; it can false-positive on a future class that happens to start with `text-` and a colour-like word. It lives in a test file and is cheap to adjust.
- The by-hand check is the only part that can't be proven by tests. Everything else here is static.
