## Context

`App.tsx` composes `ThreadPrimitive`, `MessagePrimitive` and `ComposerPrimitive` with ~40 lines of plain CSS in `index.css`. Colours are tokens (`--bg`, `--fg`, ...) guarded by `design-tokens.test.ts` (no literals outside the token blocks; >= 4.5:1 for listed pairs in both themes). Frontend guide §4.1 says "primitives, not a styled kit" and §6 "no CSS framework yet". Tests run in Vitest/Node with `renderToStaticMarkup`, no jsdom (§7.1). Current deps: `@assistant-ui/react` ^0.15.36, React 19, Vite.

The docs site `assistant-ui.com` was unreachable during triage (TLS error). Verified from npm only: the CLI package `assistant-ui` 0.0.136 exists; `@assistant-ui/react-markdown` 0.14.26 peers on `@assistant-ui/react ^0.15.0`. Everything else about the kit below is UNVERIFIED until task 1.1.

## Decisions

### D1. Adopt the assistant-ui prebuilt styled components

- **Choice**: Generate the kit's thread components with its CLI into `frontend/src/components/` and compose them in `App.tsx`.
- **Alternatives**: (a) keep primitives and hand-wire the missing features with our CSS: fewer deps, more bespoke code; the issue owner chose the kit in triage. (b) another chat UI library: would replace the runtime and adapter too.
- **Why**: Gets empty state, suggestions, action bar, stop, scroll-to-bottom and markdown as maintained code.
- **ADR**: `docs/adr/0003-styled-chat-kit.md` (Proposed), overriding frontend guide §4.1 and §6.

### D2. Tailwind + shadcn setup as the kit requires

- **Choice**: Whatever Tailwind major and shadcn config the kit CLI requires, recorded by the spike.
- **Alternatives**: Port the kit's styles to plain CSS: defeats D1.
- **Why**: The kit is Tailwind-class based.
- **UNVERIFIED** - confirm by task 1.1: Tailwind version (v3 or v4), whether `components.json` and an `@/` path alias are required, the exact install command, and the full list of added packages. If the alias is required, add it to both `tsconfig.app.json` and `vite.config.ts`.
- **ADR**: 0003.

### D3. shadcn default theme, dark mode by OS

- **Choice**: Keep the kit's default token values. Dark mode follows `prefers-color-scheme`; no toggle.
- **Alternatives**: Map our existing palette onto the kit's tokens: rejected by the issue owner (wants defaults). A class-based `.dark` toggle: out of scope.
- **Why**: Least custom theming.
- **UNVERIFIED** - confirm by task 1.1: shadcn defaults commonly switch dark mode with a `.dark` class; if so, bind the Tailwind dark variant and the dark token block to `@media (prefers-color-scheme: dark)` instead.
- **Contrast**: the default values are not assumed to pass. Task 2.2 rewrites the token test; any pair below 4.5:1 is adjusted in the token block with the measured ratio in a comment (guide §6 rule kept).
- **ADR**: 0003.

### D4. Citations stay in `metadata.custom`

- **Choice**: The generated assistant message renders our `Citations` component, reading `message.metadata.custom.citations` exactly as today. `chat/adapter.ts` and `chat/sse.ts` do not change.
- **Alternatives**: assistant-ui `source` parts: cannot carry the locator or quoted text (guide §2.2).
- **Why**: No wire or adapter change; abstentions still show no sources.
- **ADR**: none (local to this change).

### D5. Markdown via `@assistant-ui/react-markdown`

- **Choice**: Use the kit's markdown text component (built on `@assistant-ui/react-markdown`) for assistant text parts, with GFM tables.
- **Alternatives**: Plain text (today): fails the markdown acceptance criterion.
- **Why**: Ships with the kit.
- **UNVERIFIED** - confirm by task 1.1 whether `remark-gfm` ships with the kit; if not, add it (tables are an acceptance criterion). Raw HTML in model output must not render as HTML; asserted by test, not assumed.
- **ADR**: none (local to this change).

### D6. Tests stay in Node without jsdom

- **Choice**: Static-markup tests (`renderToStaticMarkup`) assert structure and accessible names for the empty state and suggestions, markdown output, citations, action-bar buttons, the composer and upload. Interactive behaviour the kit owns (stop, copy, regenerate, scroll) is checked by a manual keyboard and mouse pass recorded in `verification.md`.
- **Alternatives**: Add jsdom and testing-library: contradicts guide §7.1; a separate decision if ever needed.
- **Why**: Keeps the existing test approach.
- **UNVERIFIED** - confirm by task 1.1 whether kit components render under `renderToStaticMarkup` inside an `AssistantRuntimeProvider`; if not, tests target our own wrappers and the manual pass covers the rest.
- **ADR**: none (local to this change).

### D7. `index.css` and the token test

- **Choice**: `index.css` becomes the Tailwind entry plus the kit's token blocks; the old class rules (`.thread`, `.message`, `.composer`, ...) are deleted. `design-tokens.test.ts` is rewritten to parse the new token format (likely `oklch()`, UNVERIFIED) and keeps both guards: no colour literals outside the token blocks, and >= 4.5:1 for every text/background pair the UI uses, in both themes.
- **Alternatives**: Drop the test: loses the accessibility guarantee.
- **Why**: Keeps WCAG AA enforced in CI through the stack change.
- **ADR**: none (local to this change).

## Interfaces

No API or stream changes. Component boundary (final names come from the generated kit):

```tsx
// App.tsx
<AssistantRuntimeProvider runtime={useLocalRuntime(chatAdapter)}>
  <Upload />  {/* restyled, same behaviour */}
  <Thread />  {/* generated kit; its assistant message renders <Citations /> */}
</AssistantRuntimeProvider>
```

```ts
// unchanged contract read by the assistant message (from chat/adapter.ts)
message.metadata.custom?.citations as Citation[]
```

## Data

none

## Risks / Open Questions

- Bundle size grows (Tailwind output, Radix, icons). Record before/after `npm run build` size in `verification.md`.
- Kit output can change between CLI versions; the generated files are committed and owned by us after generation.
- Generated code may raise oxlint findings; fix them, or disable narrowly per file with a reason.
- OPEN QUESTION: the wording of the suggested questions. Default: fixed, generic, corpus-agnostic examples; confirm during task 3.2.
