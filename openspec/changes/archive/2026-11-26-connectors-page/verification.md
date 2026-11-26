## Checks Run

All run in this session, on branch `feat/34-connectors-page` (23 commits ahead of `main`) at `d1aad67`, working tree clean. This is the third run. Run 1 was FAIL: two scenarios had no test (task 8.1 added them). Run 2 was PASS. After `rag-review` raised one non-blocking finding (unused `refresh` in `useQuality`), task 8.2 removed it, and this run re-checks the final commit.

| Command | Result |
| --- | --- |
| `cd backend && uv run ruff check .` | pass (All checks passed) |
| `cd backend && uv run ruff format --check .` | pass (62 files already formatted) |
| `cd backend && uv run python -m pytest -q -rs` | pass: 215 passed, 1 skipped (`tests/test_models.py:66`, needs Ollama on localhost:11434) |
| `cd frontend && npm run lint` (oxlint) | pass, no warnings |
| `cd frontend && npm run build` (`tsc -b && vite build`) | pass (only the existing chunk-size notice) |
| `cd frontend && npm test` (from `D:\`, upper-case drive) | pass: 27 files, 169 tests |
| `python scripts/check_links.py` | pass (links ok, 138 files; the count includes this file) |
| `openspec validate connectors-page` | pass (change is valid) |
| `openspec validate --all` | pass (13 passed, 0 failed) |

Notes on the commands:
- `uv run pytest` fails on this machine with "uv trampoline failed to canonicalize script path". I used `uv run python -m pytest`; same tests, same result. This is a local setup problem, not a test failure.
- The db-marked tests (`test_runner.py` and others) ran against the local Postgres and were not skipped; the single skip is the Ollama one.
- Drift check: **not applicable**. The issue has no `db-change` label and the change adds no Alembic revision (the migrations were applied to the local dev database only to run the app; they are not part of this change).
- Not run: the Docker stack and `scripts/smoke.sh` (need models and a running compose stack); CI runs are pending until the PR is updated.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| app-shell: Open Documents from the sidebar | `shell/Sidebar.test.tsx` "marks Documents as the current page only on that page"; `shell/SidebarDrawer.test.tsx` "a plain Documents click navigates and then closes" |
| app-shell: Open Connectors and quality from the sidebar | `shell/Sidebar.test.tsx` "makes Connectors and quality the one link to the page, current only there"; `shell/AppShell.test.tsx` "shows the Connectors page, with the chat mounted but hidden…" |
| app-shell: Quality opens its dialog (now: no Quality item, no dialog) | `shell/Sidebar.test.tsx` "lists logo, New chat, Knowledge and Docs in order", "has no quality dialog any more", "has no quality state in the sidebar" |
| app-shell: Back returns to the chat | `shell/route.test.ts` "navigates to connectors and back to chat on popstate" |
| app-shell: Shared link in production | `tests/test_ui_mount.py::test_app_path_serves_index` (now requests `/app/documents` and `/app/connectors`, each 200 with `index.html`) |
| app-shell: API paths are not affected | `tests/test_ui_mount.py::test_api_and_unknown_paths_are_not_index` (now also `GET /connectors`: the JSON list with `pdf` and `docx`, no `<title>`) |
| connectors-page: Built-in connectors | `tests/test_connectors_api.py::test_built_ins_listed_sorted_with_allowlisted_fields` |
| connectors-page: Third-party connector | `tests/test_connectors_api.py::test_third_party_connector_is_not_builtin` |
| connectors-page: Failed third-party connector | `tests/test_connectors_api.py::test_failed_connectors_not_listed` |
| connectors-page: Cards for each connector | `connectors/ConnectorsPage.test.tsx` "shows a card per connector with its extensions and document count", `countByConnector` |
| connectors-page: Page frame | `connectors/ConnectorsPage.test.tsx` "has the title, intro and the 0.80 explainer with its link, in every state", "ends with Build your own…", "shows no planned connectors, and no card for a quality entry that is not loaded" |
| connectors-page: Subtitle | `connectors/ConnectorCard.test.tsx` "shows name and the subtitle: Built-in, extensions, document count" |
| quality-panel: Both connectors passed | `connectors/ConnectorCard.test.tsx` "shows Passes with three metrics, bars, the run date and model ids"; `ConnectorsPage.test.tsx` "joins quality by connector name" |
| quality-panel: A gated metric failed | `connectors/ConnectorCard.test.tsx` "shows Below bar and names only the failed metric" |
| quality-panel: Question counts | `connectors/ConnectorCard.test.tsx` "says how many questions it was checked on, only when both counts exist" |
| quality-panel: Summary without counts | same card test; `tests/test_quality.py::test_summary_without_counts_still_measures`; `tests/test_quality_api.py::test_old_summary_has_no_counts` |
| quality-panel: No valid run | `connectors/ConnectorCard.test.tsx` "shows Not measured with the command and no numbers…"; `ConnectorsPage.test.tsx` "shows Not measured on every card…"; `tests/test_quality.py::test_not_measured` (exit2, missing, malformed) |
| quality-panel: Sampled run | `tests/test_quality.py::test_not_measured[sampled]` (backend gives `not_measured`) plus the card tests above (the card renders `not_measured` as Not measured) |
| quality-panel: Different configuration | `connectors/ConnectorCard.test.tsx` "says the result is not current when the configuration differs" |
| quality-panel: Themes and keyboard | `design-tokens.test.ts` (4.5:1 pairs, light and dark); `shell/a11y.test.tsx` ("the loaded Connectors page", "keyboard reach on the Connectors page", "Connectors page sources"). The by-hand check was done by the user after `rag-review` (see Manual check below). |
| quality-panel: Summary fields only | `tests/test_quality_api.py::test_measured_has_summary_fields_only`, `test_question_counts_are_in_the_entry_and_nothing_else_new` |
| quality-panel: Counts passed through | `tests/test_quality.py::test_question_counts_pass_through`; `tests/test_quality_api.py::test_question_counts_are_in_the_entry_and_nothing_else_new` |
| quality-panel: Counts absent or invalid | `tests/test_quality.py::test_invalid_count_leaves_both_out_and_stays_measured` (five invalid cases) |
| quality-panel: Counts in the summary | `tests/test_runner.py::test_summary_is_small_and_shareable` |

## Manual check

After `rag-review` raised the REVIEW.md rule "Look at it", the dev stack was run (`npm run dev`; backend restarted on commit `b43045d` with a demo eval summary: pdf below the bar on correctness with 30 and 10 question counts, docx passing, models differing from the running configuration). The user opened `/app/connectors` and reported "all working". That check was made at commit `b43045d`; the later commit `d1aad67` (task 8.2) only removes unused code from `useQuality` and changes nothing visible. Light and dark themes, keyboard order and phone width were on the checklist given to the user; the report did not itemise them. The "Not measured" card state was not viewed in the browser (static tests only). The demo summary is synthetic, not eval output.

## Eval Results

N/A per eval-plan. The runner's summary output gains two count fields; scoring, gating, retrieval and prompts are untouched, so no eval run is required. The new fields are covered by the runner's own test. Not verified: the count fields on real model output (no eval run with models was made).

## Deviations

From design and specs, all decided in the session (issue #34 and the OpenSpec change were updated first):
- Single sidebar item "Connectors and quality"; no Quality item, Insights group or sidebar state (supersedes the first plan, tasks 3.3, 4.1 and 4.2 text).
- Page follows the app-shell design's layout and copy, on the repo's colour tokens, not the design's hex colours or blue/orange status colours.
- Cards show the upper-cased name (`PDF`, `DOCX`); the design shows "Word", and the API sends no display name.
- The failed metric's bar fill is `bg-foreground`, not red: `--destructive` on `--muted` is 4.37:1 in light mode, under 4.5:1. The pill outline, the word and the "(below 0.80)" text carry the failure.
- The card footer keeps the embedding model and the abstain threshold (the spec requires them); the design shows only chat and judge.
- The design's "Build your own" sentence ("Each must pass the conformance kit and the 0.80 bar") was reworded to "To be listed as supported, one must pass…", because the connector guide (§5) makes those the conditions for "supported", not for loading.
- Design elements not built: Planned connector cards, Unanswered and Settings pages, chat list, model status.
- Two existing checks were narrowed, not removed: `test_runner.py` (`"answer"` is now the quoted key `"answer"`, since the new `answerable_questions` keys contain the word), and `ConnectorCard.test.tsx` "no numbers" (now checks visible text, since SVG icon paths hold digits).
- `openspec/changes/connectors-page/plan.md` (the working plan file) was committed from task 5.1 on; before that it was untracked.
- The two absolute GitHub links on the page ("How it's measured", "Connector guide") are not covered by the link checker and were not checked to resolve.
- The explainer's "A connector below it still works, but its answers are less reliable" is the design's wording, kept verbatim; it is a product claim awaiting your review.

## Verdict

**PASS**

Every command passed, every spec scenario maps to a test, and eval is N/A per eval-plan. Caveats that do not change the verdict:
- The by-hand check was done by the user, and the result is recorded under Manual check. It is a one-line report, not an itemised record.
- The count fields were checked by the runner's own test, not on real model output.
- The two absolute GitHub links on the page are not covered by the link checker.
- The explainer's wording ("still works, but its answers are less reliable") awaits your product review.
- During this session the working tree was briefly switched to another branch (`feat/48-mkdocs-netlify-docs`) and back; this run is on `feat/34-connectors-page` at `d1aad67`.
