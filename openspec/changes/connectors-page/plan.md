## Task
6.2 `connectors/ConnectorCard.tsx` redesign: icon, name, subtitle "Built-in · .pdf · N documents", status pill (Passes / Below bar / Not measured, icon plus text), metric rows, footer with date, question counts when present, chat and judge models; keep Not measured with the eval command and the not-current note; existing tokens only; tests for each state

## Scope
Layout and copy from the design (board "Connectors and quality", card section), on the app's colour tokens. Same props and same pure component; only its markup changes.
- **Header row:** a 40px rounded square with a `FileTextIcon` (`aria-hidden`), then the connector name (upper-case, as now: the design shows "PDF" and "Word", but the API sends `pdf` and `docx` only, so the card keeps the upper-cased name and does not invent display names), the subtitle, and the status pill at the right (wraps below on narrow cards).
- **Subtitle:** `Built-in · .pdf · 3 documents`: "Built-in" only when `connector.builtin`, then the extensions joined with ", ", then the count ("1 document", "N documents").
- **Status pill:** outlined, no fill, icon plus word, so it uses text-on-background pairs already measured: Passes (`text-foreground`, `border-border`, check icon), Below bar (`text-destructive`, `border-destructive`, x icon), Not measured (`text-muted-foreground`, `border-border`, help icon). It replaces the separate "Passes"/"Below bar" line.
- **Metric rows (measured only):** per metric a row with the label on the left (muted) and the value in bold on the right, with "(below 0.80)" after the number for a failed metric; under it the bar: track `bg-muted`, fill `bg-foreground` (passing) or `bg-destructive` (failed), and the 0.80 mark. Bars stay `aria-hidden`; the number is text. The label and value are now two elements, so the markup reads "Correctness" then "0.70 (below 0.80)" (screen readers read them in order).
- **Not-current note:** unchanged text and `role="status"`, above the footer.
- **Footer (measured only), top border, small muted text:** "Last checked <date>" (e.g. "25 Nov 2026", via `toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })` inside the existing `<time dateTime>`), then " on A answerable and U unanswerable questions" only when both `answerable_questions` and `unanswerable_questions` are numbers, then " · Chat X · Embeddings Y · Judged by Z · Abstain threshold T". The design's footer shows only the chat and judge models; the spec (quality-panel, "Quality panel shows the latest eval result") still requires the embedding model and the threshold, so I keep them. 
- **Not measured card:** header, subtitle and the "Not measured" pill, then "Run `cmd` to measure." (same command logic as now). No metrics, no footer, no numbers.
- Guides read: `docs/guides/frontend-development-guide.md` (5 accessibility: state as words next to an `aria-hidden` icon, mismatch `role="status"`; 6 styling: tokens only, no raw colours or palette classes, 4.5:1 pairs in both themes; 7.1 static-markup tests).

## Not in scope
- The page frame, grid, intro, explainer and "Build your own" card (6.3).
- Sidebar (6.4). Guide text (7.1).
- New colour tokens, blue/orange from the design, or any display-name mapping.

## Files
| File | Change |
| --- | --- |
| `frontend/src/connectors/ConnectorCard.tsx` | new markup as above |
| `frontend/src/connectors/ConnectorCard.test.tsx` | update and extend |
| `frontend/src/connectors/ConnectorsPage.test.tsx` | the "Correctness 0.70 (below 0.80)" assertion follows the two-element markup |
| `frontend/src/design-tokens.test.ts` | add the pair `['--destructive', '--muted']` (bar fill on its track) with a comment |

## Tests
Static markup, as now:
- Subtitle (spec "Subtitle"): "Built-in · .pdf · 3 documents"; singular and zero; for `builtin: false` the word "Built-in" is absent and the subtitle is ".pdf · 3 documents".
- Passes (spec "Both connectors passed"): the pill says "Passes" with an `aria-hidden` icon; three metric labels and numbers; no "Below bar".
- Below bar (spec "A gated metric failed"): "Below bar", and "0.70 (below 0.80)" only on the failed metric; its fill uses the destructive class, passing metrics use the foreground fill.
- Footer (spec "Question counts"): with both counts: "on 30 answerable and 10 unanswerable questions"; without (spec "Summary without counts"): no "answerable" anywhere and no stray "on"; with only one count: omitted. The date, chat, embedding and judge ids and the abstain threshold are present; no raw `FAIL`/`PASS`.
- Not measured (specs "No valid run", "Sampled run"): pill "Not measured", the command, no numbers, no footer; and the fallback command when the summary lacks this connector.
- Not current (spec "Different configuration"): `role="status"` note names the differing items; absent when the config matches.
- Not by colour alone: each of the three states has an `aria-hidden` icon plus its word; bars `aria-hidden`; values in text.
- `design-tokens.test.ts`: the added pair passes in both themes. UNVERIFIED until run: if `--destructive` on `--muted` is under 4.5:1 in dark mode, I'll switch the failed fill to `bg-foreground` (the "(below 0.80)" text and the pill already carry the meaning) and drop the pair, not add a token.

## Checks to run
- From `D:\PROJECTS\Plumbline\frontend`: `npm run lint && npm run build && npm test`
- By eye (I will ask you to open `/app/connectors` in the running app, restarted with a fixture summary): card layout, wrapping at narrow width, light and dark.

## Risks / Questions
- The design's colour language (blue bars, orange pill) is not reproduced; status reads through icon, word, outline colour and the red "(below 0.80)" text. If that looks too plain next to the design, the options are new tokens (a scope change) or leaving it.
- Upper-case connector names ("DOCX") differ from the design's "Word". Mapping names would need data the API doesn't send; left as is.
- I can't check the layout without a browser; unit tests only see markup.
