# Keyboard-only walk-through (task 4.2)

Run in a browser against `npm run dev` (backend running), keyboard only. Fill
the Result column with PASS, or FAIL plus what you saw.

| # | Step | Expected | Result |
| --- | --- | --- | --- |
| 1 | Tab from the page start | Collapse button, New chat, Documents, Quality, Docs link, then the chat input, in that order. Connectors is skipped (not a control). | PASS (reported by the owner) |
| 2 | Enter on Documents | Documents page shows, URL is `/app/documents`, the link is announced as the current page. | PASS (reported by the owner) |
| 3 | Browser Back | The chat returns, unchanged. | PASS (reported by the owner) |
| 4 | Documents page, Tab | File input, Search, Type, each Delete button, with a visible focus ring on each. | PASS (reported by the owner) |
| 5 | Type in Search, change Type | The list narrows; "No documents match." appears when nothing matches. | PASS (reported by the owner) |
| 6 | Delete, Enter | The confirm dialog takes focus; Escape cancels; focus returns to the Delete button. | PASS (reported by the owner) |
| 7 | Upload a file by keyboard | The outcome is announced in the status region. | PASS (reported by the owner) |
| 8 | Quality, Enter | The dialog opens; Escape closes; focus returns to Quality. | PASS (reported by the owner) |
| 9 | Collapse, reload | Still collapsed; every icon-only item still has a name (accessible-name inspector or screen reader). | PASS (reported by the owner) |
| 10 | Window below 768 px | Only the menu button shows; Enter opens the drawer; focus is trapped; Escape closes it and focus returns to the menu button; choosing Documents closes it. | PASS (reported by the owner) |
| 11 | New chat after an answer | An empty welcome thread, also while an answer is streaming. | PASS (reported by the owner) |
| 12 | Light and dark OS theme | All text is readable; the focus ring is visible in both. | PASS (reported by the owner) |

## Automated evidence (already run)

- `shell/a11y.test.tsx`: every control on the chat and Documents pages has an
  accessible name, expanded and collapsed; `aria-current="page"` appears once,
  on Documents only; no `outline-none` lacks a `focus-visible:` replacement; no
  colour literals or palette classes in the shell or Documents page.
- Contrast: every text/background pair in the new files is an existing pair
  in `design-tokens.test.ts`, so no new pair was added.
- Information only: `--ring` against `--background` is about 2.6:1 in light
  mode and 4.2:1 in dark mode at full opacity (computed from the token
  lightness; the `ring/50` utility is lighter). Below the 3:1 WCAG 1.4.11
  guideline in light mode. The spec does not require it, and `--ring` is a
  shared token, so it is left unchanged. OPEN QUESTION for the owner.
