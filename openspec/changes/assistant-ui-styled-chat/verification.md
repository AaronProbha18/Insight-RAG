## Checks Run

Run on 2026-11-23, branch `feat/3-assistant-ui-styled-chat` at `812342f`. The global npm on the dev machine is broken (`Cannot find module 'walk-up-path'`), so the frontend checks ran the same binaries the npm scripts call.

| Command | Result |
| --- | --- |
| `node_modules/.bin/oxlint` (= `npm run lint`) | pass, exit 0, no warnings |
| `node_modules/.bin/tsc -b && node_modules/.bin/vite build` (= `npm run build`) | pass; JS 1,084.66 kB (330.59 kB gzip), CSS 40.94 kB (7.93 kB gzip) |
| `node node_modules/vitest/vitest.mjs run` (= `npm test`) | pass, 10 files, 30 tests |
| `uv run ruff check .` (backend) | pass, "All checks passed!" |
| `uv run ruff format --check .` (backend) | pass, 49 files already formatted |
| `uv run pytest -q` (backend) | pass, 126 passed, 1 skipped |
| `sh scripts/smoke.sh` (first run) | **FAIL** at chat: `INSIGHT_RAG_EMBEDDINGS is not set`. The script runs its own `docker compose up --build`, which recreated the app without the demo model settings (there is no `.env`). Environment, not code. |
| `COMPOSE_FILE="docker-compose.yml;<scratchpad>/compose.demo.yml" sh scripts/smoke.sh` | pass: health, upload, chat answered with citation "Returns Policy / General Return Window" |
| `openspec validate assistant-ui-styled-chat` | pass, valid |
| Manual pass, task 6.1 (user, http://localhost:8000, Ollama `llama3.1:8b` + `nomic-embed-text`) | pass on all items: suggestion send, Stop keeps partial text, markdown + expandable sources, Copy, Regenerate, scroll-to-bottom, upload success and red error, abstention without sources, dark mode, 375px width, keyboard-only with visible focus |

Bundle before the change (task 1.1 baseline): JS 761.99 kB (226.81 kB gzip), CSS 1.79 kB. After: +322.67 kB JS (+103.78 kB gzip), +39.15 kB CSS.

## Scenario Coverage

| Spec scenario | Test |
| --- | --- |
| Empty thread shows welcome and suggestions | `App.test.tsx` "shows a welcome and suggested questions on an empty thread" |
| Selecting a suggestion sends it | **No automated test.** Manual 6.1 only (runtime behaviour, D6) |
| Markdown is formatted | `markdown-text.test.tsx` "formats lists, code blocks and tables" |
| Raw HTML is not rendered | `markdown-text.test.tsx` "does not turn raw HTML into elements" |
| User stops streaming | **No automated test.** Manual 6.1 only; a read-only thread cannot show Stop (D6 finding, task 3.5) |
| Copy an answer | Partial: `thread.aui.test.tsx` "offers Copy and Regenerate on a completed answer" proves the control and its name; clipboard behaviour manual 6.1 only |
| Regenerate an answer | Partial: same test proves the control; re-send and replace manual 6.1 only |
| User scrolled up during streaming | Partial: `thread.aui.test.tsx` "shows scroll-to-bottom and Send…" proves the control; no-jump behaviour manual 6.1 only |
| Answer with citations | `Citations.test.tsx` "renders title, locator and cited text for each citation" |
| Abstention shows no sources | `Citations.test.tsx` "renders nothing for an empty list" |
| Upload outcome is announced | Partial: `Upload.test.tsx` "has a labelled file input and a live status region" proves the region; announced text manual 6.1 only |
| Phone width in either theme | Partial: `design-tokens.test.ts` AA for 10 pairs, both themes; no horizontal scroll at 375px manual 6.1 only |
| Keyboard-only use | Partial: accessible names in `App.test.tsx` and `thread.aui.test.tsx`; keyboard operation and visible focus manual 6.1 only |

Uncovered by any automated test: **2 scenarios** (Selecting a suggestion sends it; User stops streaming). Six more are covered only in part, with the behavioural half manual. This follows design D6 (Node-only static-markup tests, no jsdom), but the verdict rule counts a scenario without a test as a gap.

Also not caught by any test: the render loop fixed in `812342f` (a selector returning a fresh `[]`). Static renders run once, so they cannot detect it; it was found in the manual pass.

## Eval Results

N/A per eval-plan (presentation-only change; adapter, wire format and backend untouched).

## Deviations

- **Kit source**: the assistant-ui CLI could not reach its registry from the dev machine (TLS fails for every command-line client). Registry items were saved from a browser and vendored from their `content` (D1); `components.json` is kept so the CLI works where the network allows.
- **Trimmed kit**: `thread.aui.tsx` keeps only parts the `/chat` stream can produce (D1, ADR 0003). 9 of the 11 registry dependencies were not taken.
- **Tasks 2.1/2.2 re-split** (approved): setup first, then the token swap with the test rewrite, because the shadcn token names collided with ours.
- **Token change**: light `--muted-foreground` darkened from `oklch(0.556 0 0)` (4.34:1 on `--muted`) to `oklch(0.54 0 0)` (4.64:1).
- **`@assistant-ui/react` bumped** ^0.15.36 → ^0.15.37 (kit requirement).
- **`AGENTS.md` updated** (stack row, layout), beyond task 5.1's listed files, approved, so it doesn't contradict the code.
- **Test approach**: `initialMessages` load after first render, so tests use `ReadonlyThreadProvider` and `TextMessagePartProvider` (D6 findings), and some behaviour is manual-only (see coverage).
- **Bug found and fixed in the manual pass**: blank screen on the first message (render loop), commit `812342f`.
- ADR 0003 is **Proposed**; accepting it is for the human reviewer.

## Coverage Decision

First verdict this session: **FAIL**, for the 2 scenarios without an automated test. The issue owner then accepted the recorded manual pass (task 6.1) as coverage for runtime-only behaviour, recorded in `design.md` D6 ("Decision (verification, 2026-11-23, issue owner)"). The 2 scenarios and the behavioural half of the 6 partial ones are covered by that manual pass, not by an automated test. Follow-up candidate: interaction tests (jsdom/testing-library), which need a change to frontend guide §7.1.

## Verdict

**PASS** with accepted manual coverage. All checks above ran this session and pass (smoke after supplying the model settings; the first failed run is recorded). No code changed between the checks and this verdict; only `design.md` and this file were edited. The 2 scenarios without an automated test are covered by the manual pass under the D6 decision above.
