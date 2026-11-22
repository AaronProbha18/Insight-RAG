## Context

`App.tsx` composes `ThreadPrimitive`, `MessagePrimitive` and `ComposerPrimitive` with ~40 lines of plain CSS in `index.css`. Colours are tokens (`--bg`, `--fg`, ...) guarded by `design-tokens.test.ts` (no literals outside the token blocks; >= 4.5:1 for listed pairs in both themes). Frontend guide §4.1 says "primitives, not a styled kit" and §6 "no CSS framework yet". Tests run in Vitest/Node with `renderToStaticMarkup`, no jsdom (§7.1). Current deps: `@assistant-ui/react` ^0.15.36, React 19, Vite.

### Spike findings (task 1.1, 2026-11-22)

- **CLI**: `npx assistant-ui@0.0.136 init` then `add thread`. It refuses to run under a coding agent unless passed `--force`. `init` does not install Tailwind or the alias; its preflight fails until both exist. Its own dependency install fails on this machine (`EALLOWSCRIPTS`: the CLI passes `--allow-scripts`); installing the same packages with plain `npm install` works.
- **Registry access**: `r.assistant-ui.com` and `ui.shadcn.com` fail TLS from every command-line tool on the dev laptop (curl, Node, npm, PowerShell; inside and outside the sandbox) but load in the browser. So `add` cannot run here. The registry items were saved from the browser into a local, git-excluded `registry/` folder. Their `content` fields are the component sources.
- **Tailwind**: v4 via `@tailwindcss/vite` and `@import "tailwindcss"` in `index.css`.
- **Alias**: `paths: {"@/*": ["./src/*"]}` in `tsconfig.json` and `tsconfig.app.json`, plus `resolve.alias` in `vite.config.ts`. `baseUrl` must not be set: the installed TypeScript rejects it (`TS5102`).
- **`components.json`** (written by `init`): style `base-nova`, base colour `neutral`, icons `lucide`, aliases `@/components`, `@/components/ui`, `@/lib/utils`, `@/hooks`.
- **Packages**: style base `cn` (shadcn's clsx + tailwind-merge replacement, published from `github.com/shadcn-ui/cn`), `class-variance-authority`, `@base-ui/react` (**Base UI, not Radix**), `lucide-react`; dev `tw-animate-css`, `shadcn` (for `shadcn/tailwind.css`). `init` also wanted `@fontsource-variable/geist`; it is not referenced by any file we keep, so it is not added. Components: `@assistant-ui/react@^0.15.37` (bump from ^0.15.36), `@assistant-ui/react-markdown@^0.14.26`, `remark-gfm`.
- **Registry tree for `thread`**: `button`, `skeleton`, and assistant-ui `attachment`, `file`, `follow-up-suggestions`, `image`, `markdown-text`, `reasoning`, `tooltip-icon-button`, `tool-fallback`, `tool-group`. `markdown-text` needs `tooltip-icon-button` and `use-copy-to-clipboard`; `tooltip-icon-button` needs `tooltip` and `button`; `button`/`tooltip` need `utils`.
- **Tokens**: `oklch()` values (`neutral.json` `cssVars`). Greys have chroma 0, so relative luminance is `L^3`. Light `--muted-foreground` (0.556) on `--muted` (0.97) is **≈ 4.34:1, below AA**; on `--background` it is ≈ 4.73:1. Dark-mode greys pass. Other pairs are measured by the rewritten test (task 2.2).
- **Dark mode**: `shadcn/tailwind.css` does not define a `dark` variant, and `shadcn init` normally adds `@custom-variant dark (&:is(.dark *))` to the app's CSS. Without that line, Tailwind v4's `dark:` variant is `prefers-color-scheme`. The kit uses `dark:` in 7 places.
- **Markdown**: `markdown-text` uses `remark-gfm` (tables) and no `rehype-raw`, so raw HTML is not rendered as elements. Task 3.3 asserts it.
- **SSR probe**: the generated `Thread` inside `AssistantRuntimeProvider` with `useLocalRuntime(chatAdapter)` renders under `renderToStaticMarkup` in Vitest/Node. The empty thread showed "How can I help you today?", `aria-label="Message input"` and `aria-label="Send message"`.
- **Bundle** (`npm run build`): before 761.99 kB JS / 226.81 kB gzip, 1.79 kB CSS. With the kit files present but not yet used by `App`: 765.63 kB JS / 227.87 kB gzip, 29.87 kB CSS. The real "after" number is measured in task 6.1.

## Decisions

### D1. Adopt the assistant-ui prebuilt styled components, vendored and trimmed

- **Choice**: Take the kit sources from the registry items (component `content`) into `frontend/src/components/` with the CLI's path mapping, and compose them in `App.tsx`. Trim `thread.aui.tsx` to what this app uses: remove attachments, files, images, reasoning, tool calls and groups, follow-up suggestions, voice and dictation, branch picker, edit composer, feedback, the "more" menu, load-earlier and history skeleton. The registry items for those parts are not taken.
- **Alternatives**: (a) keep primitives and hand-wire the missing features with our CSS: fewer deps, more bespoke code; the issue owner chose the kit in triage. (b) another chat UI library: would replace the runtime and adapter too. (c) take the full tree: 9 more registry items for features our backend never sends.
- **Why**: Gets the empty state, suggestions, action bar, stop, scroll-to-bottom and markdown as maintained code. Trimming keeps only code paths our `/chat` stream can reach. The files are ours after vendoring either way.
- **How the sources arrive**: the CLI cannot reach the registry from the dev laptop (spike). Tasks 2.1 and 3.1 write files from the saved `registry/*.json` items; on a machine with access, `npx assistant-ui@0.0.136 add <item>` produces the same files.
- **ADR**: [0003](../../../docs/adr/0003-styled-chat-kit.md) (Proposed), overriding frontend guide §4.1 and §6.

### D2. Tailwind v4 + shadcn `base-nova` setup

- **Choice**: `tailwindcss` + `@tailwindcss/vite`; `@/*` alias in both tsconfigs (no `baseUrl`) and `vite.config.ts` (use `import.meta.dirname`, not `__dirname`, which Vite warns about); `components.json` as written by `init`; the style-base packages listed in the spike findings, without `@fontsource-variable/geist`.
- **Alternatives**: Port the kit's styles to plain CSS: defeats D1.
- **Why**: The kit is Tailwind-class based, and `components.json` lets the CLI add items later on a machine that can reach the registry.
- **ADR**: [0003](../../../docs/adr/0003-styled-chat-kit.md).

### D3. shadcn default (`neutral`) theme, dark mode by OS

- **Choice**: The `neutral` `cssVars` go in `index.css`: `:root` for light, `@media (prefers-color-scheme: dark) { :root { ... } }` for dark. No `@custom-variant dark` line, so Tailwind's `dark:` follows the OS (spike). No toggle.
- **Alternatives**: Map our existing palette onto the kit's tokens: rejected by the issue owner (wants defaults). A class-based `.dark` toggle: out of scope.
- **Why**: Least custom theming.
- **Contrast**: defaults are kept except where a pair the UI uses measures below 4.5:1. Known: light `--muted-foreground` on `--muted` ≈ 4.34:1. Task 2.2 darkens the light `--muted-foreground` to the nearest passing L and records the measured ratio in a comment (guide §6 rule kept).
- **ADR**: [0003](../../../docs/adr/0003-styled-chat-kit.md).

### D4. Citations stay in `metadata.custom`

- **Choice**: The generated assistant message renders our `Citations` component, reading `message.metadata.custom.citations` exactly as today. `chat/adapter.ts` and `chat/sse.ts` do not change.
- **Alternatives**: assistant-ui `source` parts: cannot carry the locator or quoted text (guide §2.2).
- **Why**: No wire or adapter change; abstentions still show no sources.
- **ADR**: none (local to this change).

### D5. Markdown via the kit's `markdown-text`

- **Choice**: The vendored `markdown-text` component (`@assistant-ui/react-markdown` + `remark-gfm`) renders assistant text parts. No `rehype-raw`.
- **Alternatives**: Plain text (today): fails the markdown acceptance criterion.
- **Why**: Ships with the kit; tables via GFM; raw HTML stays text (spike). Task 3.3 asserts both.
- **ADR**: none (local to this change).

### D6. Tests stay in Node without jsdom

- **Choice**: Static-markup tests (`renderToStaticMarkup`) assert structure and accessible names for the empty state and suggestions, markdown output, citations, action-bar buttons, the composer and upload. The spike confirmed the kit's `Thread` renders this way with `useLocalRuntime(chatAdapter)`. Interactive behaviour the kit owns (stop, copy, regenerate, scroll) is checked by a manual keyboard and mouse pass recorded in `verification.md`.
- **Alternatives**: Add jsdom and testing-library: contradicts guide §7.1; a separate decision if ever needed.
- **Why**: Keeps the existing test approach.
- **ADR**: none (local to this change).

### D7. `index.css` and the token test

- **Choice**: `index.css` holds `@import "tailwindcss"`, `@import "tw-animate-css"`, `@import "shadcn/tailwind.css"`, the `@theme inline` mapping, the two `neutral` token blocks and the `@layer base` rules from `index.json`. The old class rules (`.thread`, `.message`, `.composer`, ...) are deleted. `design-tokens.test.ts` is rewritten to parse `oklch()` tokens and keeps both guards: no colour literals outside the token blocks, and >= 4.5:1 for every text/background pair the UI uses, in both themes.
- **Alternatives**: Drop the test: loses the accessibility guarantee.
- **Why**: Keeps WCAG AA enforced in CI through the stack change.
- **UNVERIFIED** - confirm in task 2.1: the exact `@theme inline` block `init` writes for Tailwind v4 was not captured, because the CLI never reached the registry step that writes CSS. Write it from the `neutral` `cssVars` keys (`--color-<name>: var(--<name>)`, `--radius-*`), then confirm that utilities like `bg-background` and `text-muted-foreground` appear in the built CSS.
- **Colour literals**: the kit classes include literals outside the tokens (`amber-*` in the trimmed-away error states; `color-mix(...)` in `--composer-bg`). Trimming removes the `amber-*` uses. The literal guard scans `index.css` only, as today; component class names are not scanned.
- **ADR**: none (local to this change).

## Interfaces

No API or stream changes. Component boundary:

```tsx
// App.tsx
<AssistantRuntimeProvider runtime={useLocalRuntime(chatAdapter)}>
  <Upload />  {/* restyled, same behaviour */}
  <Thread />  {/* vendored kit; its assistant message renders <Citations /> */}
</AssistantRuntimeProvider>
```

```ts
// unchanged contract read by the assistant message (from chat/adapter.ts)
message.metadata.custom?.citations as Citation[]
```

## Data

none

## Risks / Open Questions

- Bundle size grows (Tailwind output, Base UI, icons). Record before/after `npm run build` size in `verification.md`; baseline above.
- Vendored code drifts from upstream; updating means re-fetching registry items and re-applying the trim.
- Vendored code may raise oxlint findings; fix them, or disable narrowly per file with a reason.
- `"use client"` directives in kit files are no-ops under Vite; left as vendored.
- OPEN QUESTION: the wording of the suggested questions. Default: fixed, generic, corpus-agnostic examples; confirm during task 3.2.
