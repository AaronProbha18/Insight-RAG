## Task
6.3 `connectors/ConnectorsPage.tsx` redesign: title "Connectors and quality", intro, 0.80 explainer with a "How it's measured" link, card grid, "Build your own" card linking to the connector guide, no Planned cards; links are `main` on `AaronProbha18/Plumbline`; keep load, error and empty states; tests

## Scope
Markup change in `ConnectorsView` (the pure part of the page); `ConnectorsPage` (the hook wrapper), props and data flow stay as built in 3.2.
- **Frame, always shown** (also while loading and on errors): `<h1>Connectors and quality</h1>`; the intro "A connector reads one file type and turns it into passages Plumbline can cite. Each one is checked against its own question set."; and the explainer line: a small `aria-hidden` vertical mark, then "The bar is 0.80 on correctness, faithfulness and abstention. A connector below it still works, but its answers are less reliable.", then a link "How it's measured" to `https://github.com/AaronProbha18/Plumbline/blob/main/evals/README.md` (opens in a new tab with `rel="noreferrer"`, like the sidebar's Docs link; visible focus ring).
- **Status line:** the existing `role="status"` region keeps the loading, error and "No connectors loaded." messages, now under the explainer.
- **Cards:** the same `ConnectorCard`s in a responsive grid (`grid-cols-[repeat(auto-fit,minmax(21rem,1fr))]`, gap 5), instead of the single-column stack; wider page container (`max-w-6xl`) to match the design.
- **"Build your own" card** after the grid, shown only when the page has loaded (no message, connectors and quality present, empty list included): a `<section aria-label="Build your own">` with the heading "Build your own", "Connectors are Python plug-ins. Each must pass the conformance kit and the 0.80 bar.", and a link "Connector guide" to `https://github.com/AaronProbha18/Plumbline/blob/main/docs/guides/connector-development-guide.md`. Its copy is the design's; I have not checked "conformance kit" against the guide (done below).
- **Not built:** the design's "Coming next" Planned cards (issue non-goal).
- Colours only through existing tokens; the card's footer-style small text uses `text-muted-foreground`. Links use `font-medium underline`, in `text-foreground`.
- Guides read: `docs/guides/frontend-development-guide.md` (1.4 real links, 3.2 generic messages, 5 accessibility, 6 tokens only, 7.1 static markup tests), and `docs/guides/connector-development-guide.md` to check the "Build your own" copy.

## Not in scope
- Card internals (6.2, done), sidebar (6.4), guide text (7.1).
- Hooks, `useConnectors`, `App`, `AppShell` wiring: unchanged.
- Verifying the two GitHub URLs resolve on the remote `main` (the link checker covers in-repo links only).

## Files
| File | Change |
| --- | --- |
| `frontend/src/connectors/ConnectorsPage.tsx` | frame, grid, "Build your own", two link constants |
| `frontend/src/connectors/ConnectorsPage.test.tsx` | update and extend |
| `frontend/src/shell/AppShell.test.tsx` | the connectors `<h1>` assertions follow the new title |

## Tests
Static markup:
- Page frame (spec "Page frame"): h1 "Connectors and quality", the intro sentence, the explainer with its link; present while loading and on an error too.
- Link targets: the "How it's measured" and "Connector guide" `href`s are exactly the two URLs; both have `rel="noreferrer"`.
- Cards (spec "Cards for each connector", "Subtitle"): one card per connector, in order, and the PDF card reads "Built-in · .pdf · 3 documents"; the docx card "0 documents". Joined quality and Not measured cases as before.
- "Build your own" (spec "Page frame"): present when loaded, with an empty list too, absent while loading and on either error; no "Planned" and no "Coming next" text anywhere.
- No card for a connector that isn't in the list: a quality entry for a name `/connectors` doesn't list renders no card (the card list comes from `/connectors` only).
- `AppShell` `page='connectors'`: shows the new h1 and keeps the chat mounted but hidden; other pages don't contain it.
- Existing a11y guards (`a11y.test.tsx`): every control named; the two external links have `href`; source scans cover this file. They must still pass.

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- `python scripts/check_links.py` (unchanged docs; sanity)
- By eye after 6.4: the whole page in the running app.

## Risks / Questions
- "Each must pass the conformance kit and the 0.80 bar" is the design's wording. I'll confirm against the connector guide while executing; if the guide says something different (for example that the 0.80 bar is not required of third-party connectors), I'll reword the sentence to match the guide and tell you, not ship a claim the repo doesn't make.
- The explainer's "still works, but its answers are less reliable" is the design's wording for a Below bar result; it is a product claim, so I'll keep it verbatim and flag it for your product review.
- The URLs point at GitHub `main`: UNVERIFIED that they resolve once published (the repo is where the sidebar's Docs link already points).
