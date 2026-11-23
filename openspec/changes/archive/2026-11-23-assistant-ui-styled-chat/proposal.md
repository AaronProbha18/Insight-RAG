Issue: #3

## Why

The chat uses only bare assistant-ui primitives with hand-written CSS, so it has no empty state, no markdown, no stop/copy/regenerate, no scroll-to-bottom, and citations read as a flat list. Adopting assistant-ui's prebuilt styled components gets those interactions without building each one by hand.

## What Changes

- Add Tailwind CSS and the shadcn setup the prebuilt assistant-ui components need (exact versions and files confirmed by a spike, task 1.1).
- Generate the assistant-ui prebuilt thread components into `frontend/src/` and replace the primitive-based markup in `App.tsx`.
- Use the shadcn **default** theme; dark mode follows the OS (`prefers-color-scheme`), no manual toggle.
- Render citations (still read from `metadata.custom`) inside the generated assistant message.
- Restyle `Upload` with the new kit; its behaviour and status announcements do not change.
- Remove `index.css` rules the kit replaces; rewrite `design-tokens.test.ts` so it still enforces WCAG AA in both themes against the new tokens.
- Draft ADR `docs/adr/0003-styled-chat-kit.md` (Proposed).
- Update the frontend guide (§1, §4.1, §6, §7, §9), the `REVIEW.md` frontend section and `docs/architecture/overview.md`.

## Non-Goals

- Backend, API, retrieval or `/chat` SSE wire-format changes.
- Thread history or multi-thread persistence.
- Auth or user accounts.
- A manual light/dark theme toggle.
- Changing which file types upload accepts or how errors map to messages.

## Capabilities

### New

- `chat-ui`: what the chat interface shows and lets a user do (empty state, markdown, stop/copy/regenerate, scrolling, citations, upload, accessibility, theming).

### Modified

- none

## Risk

| Area | Yes/No | Note |
| --- | --- | --- |
| Answer quality (retrieval, ranking, chunking, embeddings, prompts, graph) | No | Presentation only; the adapter and wire format are untouched. |
| Connector / ingestion | No | Upload's API call and outcome mapping stay in `upload/api.ts`. |
| Auth / document ACLs | No | None exist; nothing added. |
| Public API or stream contract | No | `/chat` and `/documents` unchanged; `chat/adapter.ts` unchanged. |
| Persistence / migration | No | No storage. |
| New external dependency or service | Yes | Tailwind, shadcn tooling, `@assistant-ui/react-markdown` and the kit's peer packages. Stack change recorded in ADR 0003; overrides frontend guide §4.1 and §6. |

Tier `rag-standard`: the stack change outlives this change and needs a design and an ADR.

## Impact

- `frontend/`: `package.json`, Vite/TS config, `index.css`, `App.tsx`, `chat/Citations.tsx`, `upload/Upload.tsx`, a new generated component folder, tests.
- `Dockerfile` frontend build stage, only if the spike shows the build needs it.
- Docs: ADR 0003, frontend guide, `REVIEW.md`, architecture overview.
