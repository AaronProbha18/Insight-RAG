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
  App.tsx              composes features; no logic
  chat/   sse.ts       SSE parser
          adapter.ts   assistant-ui adapter for POST /chat
          Citations.tsx
  upload/ api.ts       POST /documents -> user-facing outcome
          Upload.tsx
```

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

**2.3 Failures throw**, so assistant-ui marks the message as failed rather
than showing partial text:

```ts
// ✅ chat/adapter.ts
    if (!response.ok || !response.body) throw new Error(`chat request failed (${response.status})`)
      } else if (event === 'error') throw new Error(data.message)
```

**2.4 The wire format changes in one PR with the backend.**
`backend/src/insight_rag/api/app.py` produces it and `chat/adapter.ts`
consumes it.

**2.5 Parse only complete SSE blocks.** An event can arrive split across
network chunks:

```ts
// ✅ chat/sse.ts
    while ((end = buffer.indexOf('\n\n')) >= 0) {
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

**4.1 assistant-ui primitives, not a styled kit.** `ThreadPrimitive`,
`MessagePrimitive` and `ComposerPrimitive` with our own CSS. A styled kit
(shadcn copy-in) can come when the design needs it.

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
  `aria-label`, the file input via its `<label>`, Send via its text, and
  each source via its `<summary>`. *Checked by* `src/App.test.tsx`.
- **Async results are announced:** the upload outcome lives in a
  `role="status"` / `aria-live="polite"` region.
- **Focus is always visible:** each control has a `:focus-visible`
  outline in `index.css`.
- **Keyboard-only use works:** upload, ask and open a source without a
  mouse. Check this by hand whenever you add a control.

## 6. Styling

- Plain CSS in `src/index.css`. There's no CSS framework yet.
- **Every colour is a token** on `:root` (`--bg`, `--fg`, `--muted`,
  `--accent`, `--border`, `--on-accent`, `--error`), redefined under
  `prefers-color-scheme: dark`. Rules use `var(--token)` only; there are
  no colour literals outside the two token blocks.
- **Each text/background pair meets WCAG AA (≥ 4.5:1) in both themes.**
  The measured ratio sits in a comment next to the token. When you add or
  change a token, measure both themes; a colour that passes in light mode
  can fail in dark mode (white on the dark-mode accent measured 2.7:1).

## 7. Testing

**7.1 Vitest in Node, without jsdom or testing-library.** Components are
stateless enough to check as static markup with React's own
`renderToStaticMarkup`:

```tsx
// ✅ chat/Citations.test.tsx
    expect(renderToStaticMarkup(<Citations citations={[]} />)).toBe('')
```

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
| Use CSS tokens | Add new hard-coded colours |
| Test with `renderToStaticMarkup` and stubbed `fetch` | Add jsdom for markup checks |
| Name files so they differ by more than case | Put `upload.ts` next to `Upload.tsx` |
