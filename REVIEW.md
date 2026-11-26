# Review checklist

Read this before reviewing a diff, whether you're a person or an AI
agent. Check the sections for the areas the diff touches. Each item points
to the rule it comes from; the guide holds the reasoning. Items marked
*(CI)* are already enforced by a test, so reviewing them means checking the
test wasn't weakened or skipped to get the change through.

## Every diff

- [ ] Every changed line traces to the task. There are no unrelated
      refactors, reformatting, or speculative abstractions
      ([CLAUDE.md](CLAUDE.md) guidelines 2–3).
- [ ] Behaviour, a contract, a boundary rule or a command changed? Then
      [the overview](docs/architecture/overview.md) and the matching guide
      changed in the same diff ([AGENTS.md](AGENTS.md) rule 7). Links still
      resolve *(CI)*.
- [ ] No secrets, real personal data, internal URLs or organisation names
      anywhere, including fixtures, eval sets and commit messages.
- [ ] Unverified library behaviour is marked UNVERIFIED, not stated as
      fact.
- [ ] A test was removed, skipped or loosened? The diff says why, and the
      reason holds.

## `backend/`

Rules: [backend guide](docs/guides/backend-development-guide.md).

- [ ] Routes stay thin. Logic lives in modules and dependencies are
      injected (§1). `graph.py` doesn't import `psycopg` or `fastapi`;
      configuration is read only by its owners *(CI: `test_boundaries.py`)*.
- [ ] Status codes follow §2.1. No internal ids, retrieval scores or
      confidence numbers in responses (§2.2–2.3) *(CI for `/chat`)*.
- [ ] Errors are domain exceptions mapped at the edge. There's no broad
      `except` in business logic, and clients get no internal details
      (§3).
- [ ] The ingestion invariants still hold: embed before writing, one
      transaction, hash idempotency, chunks inside sections (§4) *(CI)*.
- [ ] **Prompt, threshold, `top_k` or chunking changed?** Then the diff
      includes an eval run against the baseline, with no golden answers
      edited to pass (§5.3). Without one, don't approve.
- [ ] `/chat` events changed? Then `frontend/src/chat/adapter.ts` changed
      in the same diff (§2.4).
- [ ] Schema changed? It's a **new** migration revision, not an edit to an
      applied one (§7).
- [ ] New code that can fail has a test for the failure path (§8.4).

## `frontend/`

Rules: [frontend guide](docs/guides/frontend-development-guide.md).

- [ ] Only `chat/adapter.ts` calls `/chat`, and only `chat/sse.ts` parses
      SSE (§2) *(CI: `wire-format.test.ts`)*.
- [ ] Colours come only from tokens, used through token utilities
      (`bg-background`, `text-muted-foreground`), never literals or palette
      classes, and every text/background pair meets 4.5:1 in **both**
      themes (§6) *(CI: `design-tokens.test.ts`)*. A new pair the UI uses
      is added to that test's `PAIRS`.
- [ ] Vendored kit code (`src/components/`) stays trimmed to what `/chat`
      can produce ([ADR 0003](docs/adr/0003-styled-chat-kit.md), §4.1).
- [ ] Every new control has an accessible name and a visible
      `focus-visible:` outline or ring, and works by keyboard (§5). Hiding or
      disabling a button **is not access control**: anything that must not
      happen is refused by the backend.
- [ ] 5xx responses show generic text, never backend details (§3.2).
- [ ] Native elements are used where they do the job: `<details>`,
      `<label>`, `<input>` (§4.2).
- [ ] No two file names in one folder differ only by case (§1.2).
- [ ] **Look at it.** Was the UI actually run (`npm run dev`) and the
      change seen in a browser, in light and dark mode? The tests check
      markup and tokens, not how it looks.

## Connectors

Rules: [connector guide](docs/guides/connector-development-guide.md).

- [ ] Uses only `plumbline.connectors.contract` *(CI for built-ins)*.
- [ ] Unreadable input raises `ParseError`, never another exception, and
      `run_conformance` passes on its fixtures plus a `corrupt/` folder.
- [ ] Locators are something a person can follow back to the source.
- [ ] "Supported" is only claimed with a golden set that passes the 0.80
      gates.
