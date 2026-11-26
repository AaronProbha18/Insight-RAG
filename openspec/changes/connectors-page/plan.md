## Task
7.1 Update the backend guide (`/connectors` has four fields; `/quality` counts), frontend guide (folder tree, sidebar and page accessibility text, checklist) and the architecture overview; remove the Quality-link wording added in 4.1

## Scope
Docs only, no code. Each edit makes a sentence true of the code now on the branch (tasks 5.1 to 6.4).

**`docs/guides/backend-development-guide.md`**
- 2.5 (`/quality` allowlist): add that each connector entry also carries `answerable_questions` and `unanswerable_questions` when the summary has both as non-negative integers (both or neither; never a reason for `not_measured`), and name the tests: `tests/test_quality.py` and `tests/test_quality_api.py` (already named) plus `tests/test_runner.py` for the runner's side.
- 2.6 (`/connectors` allowlist): "three named fields" becomes four: `name`, `extensions`, `contract_version` and `builtin` (true when the class's module is under `plumbline.connectors.builtin`, so the built-in `pdf` and `docx` are true and a third-party connector is false).

**`docs/guides/frontend-development-guide.md`**
- 1.1 folder tree: `Sidebar.tsx` line becomes "navigation: New chat, Documents, Connectors and quality, docs link"; `ConnectorCard.tsx` says "one connector: subtitle, status pill, metric bars, last-checked footer"; `ConnectorsPage.tsx` says "title, intro, 0.80 explainer, card grid and a Build your own card; joins /connectors, /quality and /documents by connector name"; `useQuality.ts` says "the latest eval result, fetched once in App and passed to the Connectors page"; delete the `label.ts` line.
- 5 accessibility, "The sidebar" bullet: one link, "Connectors and quality", to `/app/connectors` with `aria-current` on that page; no Quality item, no Insights group and no quality state in the sidebar; drop the four "Quality: ..." wordings.
- 5 accessibility, "The Connectors page" bullet: remove `quality/label.test.ts` from *Checked by*; add that the two external links ("How it's measured", "Connector guide") are real `<a>` elements with a visible focus ring and `rel="noreferrer"`; that "Build your own" appears only once the page has loaded.
- 7.1 by-hand checklist: "Tab reaches the Connectors and quality link" (was "the Connectors and Quality links").
- 6 styling: no change needed (the page uses tokens only; no new pair).

**`docs/architecture/overview.md`**
- "Where new work goes", quality row: "the Connectors page reads it" (remove "and the sidebar's Quality label").
- "List connectors" flow: each entry has `name`, `extensions`, `contract_version` and `builtin` (true for connectors shipped in this repo, derived from the class's module).
- "Show the last eval" flow: add that each connector entry carries the answerable and unanswerable question counts when the summary has them; the runner writes them (`evals/runner.py`), and a summary from before that has none, so the page omits them.
- "Open a page" flow: no change needed (already says "Connectors at `/app/connectors`" and the client-side join).

Guides read: all three docs; AGENTS.md workflow rule 7.

## Not in scope
- Code, tests, specs, ADRs, README, AGENTS.md.
- `evals/README.md` (its sentence about the counts was added in 5.2).
- The accessibility re-check itself (7.2) and `verification.md`.

## Files
| File | Change |
| --- | --- |
| `docs/guides/backend-development-guide.md` | 2.5 and 2.6 |
| `docs/guides/frontend-development-guide.md` | tree, sidebar and page bullets, checklist |
| `docs/architecture/overview.md` | quality row, two flows |

## Tests
- None new. Docs are checked by the link checker and by greps for stale wording.
- Read-through: no remaining mention of the sidebar's Quality item or label, `quality/label`, "Insights", or "three named fields" in `docs/`.

## Checks to run
- `grep -rn "sidebar's Quality\|Quality link\|Connectors and Quality\|label\.ts\|label\.test\|Insights\|three named fields\|\"Quality: " docs README.md AGENTS.md` returns nothing stale
- `python scripts/check_links.py`
- `openspec validate --all`
- `cd backend && uv run python -m pytest tests/test_check_links.py -q`

## Risks / Questions
- None. The two external GitHub links on the page are not covered by the link checker (said before), so the guide only states that they exist.
