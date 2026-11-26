## Task
4.2 Accessibility pass: labels not by colour alone, 4.5:1 contrast in both themes (extend `design-tokens.test.ts` for new pairs), keyboard reach for every link

## Scope
This is an audit with guards, not new UI. Findings from reading the code on this branch:

- **Contrast:** the Connectors page and card use only existing tokens, and each text/background pair they produce is already in `PAIRS` in `design-tokens.test.ts`:
  - `text-foreground` on `--background` (card body, h1);
  - `text-muted-foreground` on `--background` (extensions line, run line, sidebar group labels);
  - `text-destructive` on `--background` (Below bar, not-current note, error messages);
  - the sidebar's current-item `bg-muted` under foreground text (`--foreground` on `--muted`);
  - metric bars: fill and 0.80 marker are `bg-foreground` on a `bg-muted` track (`--foreground` on `--muted`, already ≥ 4.5:1, so it also clears the 3:1 non-text rule).
  So **no new token and no new pair**. I'll document that in `PAIRS` with comments naming the new users (no assertion changes), rather than add a duplicate pair. If you want an explicit bar pair line anyway, it's one line.
- **Not by colour alone:** already structural in 3.1 and 3.3 (word plus `aria-hidden` icon, bar `aria-hidden`, value as text); I add a guard that spans the whole page, below.
- **Keyboard:** every new link is a native `<a href>` (Connectors, Quality), so it's in the tab order with the existing `focus-visible` ring from the sidebar's `item` class; nothing uses `tabindex` or a click-only element. The cards contain no interactive controls.

Changes:
1. `shell/a11y.test.tsx`: add `['connectors', false]` and `['connectors', true]` to the `describe.each` (every control named; the "announces the current page" check becomes: Documents page has 1 `aria-current`, Connectors page has 2 — the Connectors and Quality links — chat has 0); add `../connectors/ConnectorCard.tsx`, `../connectors/ConnectorsPage.tsx` to the `files` list so the outline-needs-focus-visible and no-colour-literal source scans cover them.
2. `shell/a11y.test.tsx`: one new test: in the connectors-page markup, every `<a>` has an `href` and none has `tabindex="-1"` (keyboard reach), and no link is `aria-disabled` (the old Coming-soon span is gone).
3. `design-tokens.test.ts`: comments on the existing `PAIRS` lines listing the Connectors page uses; no behaviour change.

Guides read: `docs/guides/frontend-development-guide.md` §5 accessibility and §6 styling (every text/background pair ≥ 4.5:1 in both themes; tokens only; keep tests next to code), §7.1 (focus order and themes are checked by hand).

## Not in scope
- New colour tokens or restyling; none needed.
- Screen-reader testing with real assistive tech.
- Guide text: §5 and §7.1 already describe the Connectors page (task 4.1).

## Files
| File | Change |
| --- | --- |
| `frontend/src/shell/a11y.test.tsx` | connectors cases, source scan files, link reach test |
| `frontend/src/design-tokens.test.ts` | comments on `PAIRS` only |

## Tests
- The new a11y cases above (spec "Themes and keyboard": Passes and Below bar not by colour alone, text meets 4.5:1, every link reachable).
- Existing `design-tokens.test.ts` already asserts all pairs in both themes; it must still pass unchanged.

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- Manual, in a browser (this is the part I can't verify from tests; I will not tick it for you): `npm run dev`, open `/app/connectors` in light and dark OS themes; Tab through the sidebar to Connectors and Quality, Enter opens the page, focus ring visible on each; read the card states. Tell me the result, or I record it as "not checked by hand" in `verification.md`.

## Risks / Questions
- The manual keyboard and theme check can't be done by static tests. The task ends with the automated guards green and the manual step either done by you or recorded as unchecked.
- Measured contrast is only as good as the existing token test's OKLCH math; I'm relying on it as for earlier pages.
