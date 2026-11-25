# Frontend development guide

Rules for the chat UI in `frontend/`, for people and AI agents alike.
Every rule describes how the code already works. The frontend is small on
purpose: it renders what the backend decides, and owns no answer logic.
Start with [the architecture overview](../architecture/overview.md); the
backend side of the `/chat` contract is in the
[backend guide](backend-development-guide.md#2-api-conventions).

## Contents

1. [Code organisation](#1-code-organisation)
2. [The chat runtime](#2-the-chat-runtime)
3. [Calling the API](#3-calling-the-api)
4. [Components](#4-components)
5. [Accessibility](#5-accessibility)
6. [Styling](#6-styling)
7. [Testing](#7-testing)
8. [Tooling](#8-tooling)
9. [Quick reference](#9-quick-reference)

## 1. Code organisation

**1.1 One folder per feature**, holding network code apart from components:

```text
src/
  App.tsx              composes Upload + Thread, welcome suggestions; no logic
  chat/   sse.ts       SSE parser
          adapter.ts   assistant-ui adapter for POST /chat
          Citations.tsx
  upload/ api.ts       POST /documents -> user-facing outcome
          Upload.tsx
  documents/ api.ts    GET /documents, DELETE /documents/{id}
          DocumentList.tsx   rows: title, type, date, delete button
          Documents.tsx      "Documents (n)" button, dialog, delete confirm
  components/
    assistant-ui/elements/  vendored kit: thread.aui.tsx, markdown-text.tsx,
                            tooltip-icon-button.tsx (ADR 0003)
    ui/                     vendored shadcn: button.tsx, tooltip.tsx
  lib/utils.ts              vendored `cn` helper
  hooks/                    vendored hooks (use-copy-to-clipboard.ts)
```

Vendored files are imported through the `@/` alias (`@/components/...`),
set in both tsconfigs and `vite.config.ts`.

**1.2 File names must differ by more than case.** `upload.ts` next to
`Upload.tsx` collides on Windows and macOS, whose filesystems ignore case,
and TypeScript then can't tell which file `./upload` means. That's why the
helper is `api.ts`.

**1.3 Tests sit next to the code they test** (`sse.test.ts` beside
`sse.ts`).

## 2. The chat runtime

**2.1 assistant-ui local runtime plus our own adapter.** The adapter is the
**only** code that knows the `/chat` wire format (design D8):

```tsx
// ✅ App.tsx
  const runtime = useLocalRuntime(chatAdapter)
```

**2.2 Citations travel in `metadata.custom`**, not in assistant-ui
`source` parts. Those can't carry a locator or the quoted text (finding
from the 2.1 spike).

```ts
// ✅ chat/adapter.ts
      yield { content: [{ type: 'text', text }], metadata: { custom: { citations, abstained } } }
```

The `done` event's `answer_id` arrives in a last update as
`metadata.custom.answerId`; it is absent on the error path, which is how the
feedback controls know not to appear.

**2.3 Failures throw**, so assistant-ui marks the message as failed rather
than showing partial text:

```ts
// ✅ chat/adapter.ts
    if (!response.ok || !response.body) throw new Error(`chat request failed (${response.status})`)
      } else if (event === 'error') throw new Error(data.message)
```

**2.4 The wire format changes in one PR with the backend.**
`backend/src/plumbline/api/app.py` produces it and `chat/adapter.ts`
consumes it.

**2.5 Parse only complete SSE blocks.** An event can arrive split across
network chunks:

```ts
// ✅ chat/sse.ts
    while ((end = buffer.indexOf('\n\n')) >= 0) {
```

**2.6 Welcome suggestions are an `AuiConfig` scope** on the provider, not
markup. They are generic so they fit any uploaded documents:

```tsx
// ✅ App.tsx
    <AssistantRuntimeProvider runtime={runtime} config={config}>
```

## 3. Calling the API

**3.1 Relative paths only** (`/chat`, `/documents`). In Docker, FastAPI
serves the UI on the same origin. In dev, Vite proxies to the backend,
which you can point elsewhere with `VITE_BACKEND`:

```ts
// ✅ vite.config.ts
const backend = process.env.VITE_BACKEND ?? 'http://localhost:8000'
```

**3.2 Map every status to a user-facing outcome.** Show the backend's
message only where it is meant for users (415 lists the supported types).
Never show 5xx details:

```ts
// ✅ upload/api.ts
    default:
      return { kind: 'error', message: 'Upload failed. Try again.' }
```

## 4. Components

**4.1 The assistant-ui styled kit, vendored and trimmed**
([ADR 0003](../adr/0003-styled-chat-kit.md)). The kit is shadcn registry
items (`base-nova` style on Base UI), copied into `src/components/`. The
copies are ours: edit them directly. `thread.aui.tsx` is trimmed to what
the `/chat` stream can produce, so there is no attachment, tool, reasoning,
voice, branch or edit UI:

```tsx
// ✅ components/assistant-ui/elements/thread.aui.tsx
// Vendored from the assistant-ui registry (base-nova `thread`) and trimmed to
```

Feedback is the exception to "kit parts only": the thumbs controls are
our own `feedback/FeedbackControls.tsx`, built from the kit's buttons and
placed in the assistant action bar, because the kit's feedback adapter
can't carry a reason or report a failed save (ADR 0003 amendment).

To add a kit part, save its registry item (`npx assistant-ui@<version> add
<item>`, or the JSON from `https://r.assistant-ui.com/...` in a browser
where the CLI can't reach it), copy each file's `content` to the path the
CLI would use, and trim it the same way. `components.json` records the
shadcn config the CLI expects.

**4.2 Native elements first.** `<details>` for expandable citations, and a
real `<input type="file">` inside a `<label>`. Both are keyboard-operable
and announced by screen readers with no custom code.

**4.3 Render from data; avoid local state.** `Citations` renders nothing
for an empty list, so abstentions show no sources without a special case:

```tsx
// ✅ chat/Citations.tsx
  if (citations.length === 0) return null
```

## 5. Accessibility

- **Every control has an accessible name:** the question box via
  `aria-label` ("Ask a question about your documents"), the file input via
  its `<label>`, icon buttons via `aria-label` or the hidden tooltip text
  ("Send message", "Stop generating", "Copy", "Regenerate", "Scroll to
  bottom"), suggestions via their text, and each source via its
  `<summary>`. *Checked by* `src/App.test.tsx` and
  `components/assistant-ui/elements/thread.aui.test.tsx`.
- **The documents dialog:** the button's text ("Documents (n)") is its name, each delete button is named `Delete <title>`, deleting asks for confirmation in an alert dialog, and load/delete failures appear in a `role="status"` region. *Checked by* `documents/DocumentList.test.tsx` and `documents/Documents.test.tsx`.
- **Feedback controls:** "Good answer" and "Bad answer" are named buttons
  with `aria-pressed`; the thumbs-down form is a native `<fieldset>` with a
  `<legend>`, radio inputs and a labelled textarea; the outcome
  ("Feedback recorded" or a generic failure) is a `role="status"` line.
  *Checked by* `feedback/FeedbackControls.test.tsx` and
  `components/assistant-ui/elements/thread.aui.test.tsx`.
- **Async results are announced:** the upload outcome lives in a
  `role="status"` / `aria-live="polite"` region.
- **Focus is always visible:** controls use `focus-visible:` outline or
  ring utilities (the kit's buttons have them built in).
- **Keyboard-only use works:** upload, ask, stop, copy, regenerate and open
  a source without a mouse. Check this by hand whenever you add a control.

## 6. Styling

- **Tailwind CSS v4** (`@tailwindcss/vite`). `src/index.css` holds the
  imports, the tokens, the `@theme inline` mapping and the base layer, not
  component rules; components style themselves with utility classes.
- **Every colour is a token**: shadcn's `neutral` set as `oklch()` values on
  `:root`, redefined under `prefers-color-scheme: dark`. `@theme inline`
  maps each to a utility (`--color-background: var(--background)`), so
  components use `bg-background`, `text-muted-foreground`, `border-border`,
  and never raw colours or palette classes like `amber-500`. There are no
  colour literals outside the two token blocks.
- **Dark mode follows the OS.** There is no `@custom-variant dark` line, so
  Tailwind's `dark:` variant is `prefers-color-scheme`. Don't add a `.dark`
  class toggle without an ADR.
- **Each text/background pair meets WCAG AA (≥ 4.5:1) in both themes.**
  An adjusted token carries its measured ratio in a comment. When you add
  or change a token, measure both themes; a default can fail:

```css
/* ✅ index.css */
  --muted-foreground: oklch(0.54 0 0); /* shadcn default 0.556 was 4.34:1 on --muted; this is 4.64:1 (5.06:1 on --background) */
```

## 7. Testing

**7.1 Vitest in Node, without jsdom or testing-library.** Components are
stateless enough to check as static markup with React's own
`renderToStaticMarkup`:

```tsx
// ✅ chat/Citations.test.tsx
    expect(renderToStaticMarkup(<Citations citations={[]} />)).toBe('')
```

`useLocalRuntime(adapter, { initialMessages })` loads messages after the
first render, so a static render of the thread shows none. Render a
populated thread with `ReadonlyThreadProvider`, or one answer with
`TextMessagePartProvider`:

```tsx
// ✅ components/assistant-ui/elements/markdown-text.test.tsx
    <TextMessagePartProvider text={text} isRunning={false}>
```

A read-only thread can't run or cancel, so Stop, Copy (the clipboard),
Regenerate and scrolling are checked by hand in a browser, and so is the
feedback form (Tab order through Copy, Regenerate, Good, Bad; Enter and
Space; arrow keys across the reasons; Cancel and Send return focus to the
Bad button; focus rings; light and dark themes) and the
documents dialog (open and close, Tab order, focus ring and focus return,
confirm and cancel, light and dark themes).

**7.2 Network code is tested with a stubbed `fetch`** that returns real
`Response` objects, so status codes and streaming bodies behave as they do
in a browser:

```ts
// ✅ chat/adapter.test.ts
  vi.stubGlobal('fetch', vi.fn(async () => new Response(sse, { status })))
```

**7.3 Test the edges:** SSE events split across chunks, `error` events,
non-200 responses, network failures and HTML escaping all have tests
today. New network or parsing code needs the same.

**7.4 On Windows, run Vitest from an upper-case drive letter** (`D:\…`).
From `d:\…` it loads itself twice, and every suite fails with "failed to
find the current suite".

## 8. Tooling

| Command | What | In CI |
| --- | --- | --- |
| `npm run lint` | oxlint | yes |
| `npm run build` | `tsc -b` typecheck + Vite build | yes |
| `npm test` | Vitest | yes |
| `npm run dev` | dev server on :5173, proxied to the backend | no |

## 9. Quick reference

| Do | Don't |
| --- | --- |
| Keep the `/chat` format inside `chat/adapter.ts` | Parse SSE in components |
| Change the wire format in one PR with the backend | Change only one side |
| Use relative API paths | Hard-code `http://localhost:8000` in `src/` |
| Show generic text for 5xx | Show backend error details to users |
| Use native `<details>`, `<label>`, `<input>` | Build custom widgets for the same thing |
| Give every control a name and a visible focus | Rely on placeholder text as a label |
| Use token utilities (`bg-background`, `text-muted-foreground`) | Add colour literals or palette classes (`amber-500`) |
| Keep vendored kit parts trimmed to what `/chat` sends | Add kit parts for features the backend lacks |
| Test with `renderToStaticMarkup` and stubbed `fetch` | Add jsdom for markup checks |
| Name files so they differ by more than case | Put `upload.ts` next to `Upload.tsx` |
