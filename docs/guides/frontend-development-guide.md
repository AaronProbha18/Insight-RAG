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
  App.tsx              wiring only: route, collapse state, documents, saved chats, which chat the pane shows
  shell/  route.ts     useRoute(), navigate(): pages over the History API
          collapsed.ts sidebar collapse, kept in localStorage
          chatTarget.ts   what the chat pane shows: syncTarget, newChat, adopt
          Sidebar.tsx  navigation: New chat, the Chats section, Documents, Connectors and quality, docs link
          SidebarDrawer.tsx  menu button + the sidebar as a drawer below md
          AppShell.tsx layout and page switch
  chat/   sse.ts       SSE parser
          adapter.ts   assistant-ui adapter for POST /chat (createChatAdapter)
          restore.ts   saved turns -> initial thread messages
          ChatHeader.tsx  conversation title and "Searching all N documents"
          ChatPane.tsx runtime, welcome suggestions and Thread
          Citations.tsx
          FirstRun.tsx first-run screen: upload link, "Try the sample documents" (trySamples)
  conversations/ api.ts   GET, PATCH, DELETE /conversations: list, get, rename, delete
          useConversations.ts  the sidebar list: fetched on mount and after refresh, a rename or a delete
          useConversation.ts   one saved chat for the page that opens it: loading, found, not-found, error
          ChatList.tsx         the sidebar Chats section: links, inline rename and delete confirm
          saveAction.ts        what Save does with an entered title
  upload/ api.ts       POST /documents -> user-facing outcome
  documents/ api.ts    GET /documents, DELETE /documents/{id}, POST /documents/samples
          useDocuments.ts    list, `loaded`, refresh and delete, shared by page and sidebar count; trySamples
          pageLogic.ts       title search, type filter, upload-then-refresh, count per connector
          DocumentList.tsx   rows: title, type, date, delete button
          DocumentsPage.tsx  upload, search, type filter, list, delete confirm
  connectors/ api.ts   GET /connectors
          useConnectors.ts   the loaded connectors, fetched when the page shows
          ConnectorCard.tsx  one connector: subtitle, status pill, metric bars, last-checked footer
          ConnectorsPage.tsx title, intro, 0.80 explainer, card grid, Build your own card;
                             joins /connectors, /quality and /documents by connector name
  quality/ api.ts      GET /quality
          useQuality.ts      the latest eval result, fetched once in App and passed to the Connectors page
  components/
    assistant-ui/elements/  vendored kit: thread.aui.tsx, markdown-text.tsx,
                            tooltip-icon-button.tsx (ADR 0003)
    ui/                     vendored shadcn: button.tsx, tooltip.tsx
  lib/utils.ts              vendored `cn` helper
  hooks/                    vendored hooks (use-copy-to-clipboard.ts)
```

Vendored files are imported through the `@/` alias (`@/components/...`),
set in both tsconfigs and `vite.config.ts`.

**1.2 File names must differ by more than case.** A file `upload.ts` beside
a component `Upload.tsx` would collide on Windows and macOS, whose
filesystems ignore case, and TypeScript then can't tell which file
`./upload` means. That's why a feature's network code is `api.ts`.

**1.3 Tests sit next to the code they test** (`sse.test.ts` beside
`sse.ts`).

**1.4 Pages are URLs under `/app/`.** The chat is at `/`, a saved chat at
`/app/chats/{id}` (`chatIdFor` reads the id; `replace(path)` changes the URL
without a history entry), Documents at `/app/documents` and Connectors at
`/app/connectors`. A client page never reuses an API path (`/documents`,
`/connectors` and `/conversations` are the API), so the prefix can't collide with a route. Routing is the built-in
`shell/route.ts`: `useRoute()` reads the path and `navigate(path)` pushes
history; back and forward work through `popstate`. There is no router
library, and one comes when pages need nested routes or loaders. Sidebar
links are real `<a href>` elements, and only a plain left-click is handled
(`isPlainClick`), so middle-click and copy-link keep working. In
production the backend serves `index.html` for `GET /app/{path}` when
`PLUMBLINE_UI_DIR` is set, so a reload or a shared page link works; paths
outside `/app/` keep their API behaviour. *Enforced by*
`backend/tests/test_ui_mount.py`.

## 2. The chat runtime

**2.1 assistant-ui local runtime plus our own adapter.** The adapter is the
**only** code that knows the `/chat` wire format (design D8):

```tsx
// ✅ chat/ChatPane.tsx
  const runtime = useLocalRuntime(adapter, { initialMessages })
```

The runtime lives in `ChatPane`. `AppShell` keeps `ChatPane` mounted but
`hidden` while another page shows, so the conversation survives
navigation.

**What the pane shows** is the `Target` in `App.tsx` (`shell/chatTarget.ts`):
a `key`, the saved chat's `id` and whether it was `adopted`. A new `key`
remounts the pane. `syncTarget` follows the URL, but only while the chat page
is open, so a visit to Documents does not reset the chat. `newChat` gives an
empty thread, and the sidebar list keeps the previous chat.
`useConversation` loads a saved chat (`/app/chats/{id}`) and `AppShell` mounts
it with its `turns`; `initialMessages` is read on mount only, which is why
switching chats means a new `key`. After the first `done` of a new chat,
`adopt` records the id under the same `key`, so the live thread stays while
the URL is replaced with `/app/chats/{id}` and the list refreshes.
`createChatAdapter(conversation)` sends the `conversation_id` it holds and
learns it from `done`; the `onConversation` callback is captured when the
pane mounts.

On the chat page, `AppShell` shows nothing until `useDocuments` reports
`loaded`, so there is no first-run flash. It then shows `FirstRun` when the
list is empty and loaded without an error, and the chat otherwise. `ChatPane`
stays mounted but hidden while `FirstRun` shows, so deleting every document
does not lose the conversation.

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

A restored answer (`chat/restore.ts`) carries the same `metadata.custom`, so
its citations and feedback controls render through the same components. Saved
citations keep only `document_id`, `title`, `locator` and `indexed`: no `id` and
no quoted `text`. `Citations` shows such a citation as `title — locator` with
nothing to expand, and adds "Source no longer indexed" when `indexed` is
false; the inline `[c3]` markers in a restored answer have no matching label.

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
// ✅ chat/ChatPane.tsx
    <AssistantRuntimeProvider runtime={runtime} config={config}>
```

After **Try the sample documents** indexes the samples (`POST
/documents/samples`, idempotent), `AppShell` passes `samples` to `ChatPane`,
which puts one question about the sample returns policy first.

## 3. Calling the API

**3.1 Relative paths only** (`/chat`, `/documents`). In Docker, FastAPI
serves the UI on the same origin. In dev, Vite proxies to the backend,
which you can point elsewhere with `VITE_BACKEND`. Vite serves
`index.html` for page URLs like `/app/documents` itself, so they need no
proxy entry. A new API route that the UI calls needs a proxy entry; a new
UI page does not (rule 1.4):

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
- **The sidebar:** a `<nav aria-label="Main">`. The current page is `aria-current="page"` on its link. Documents is a real link whose text includes the count ("Documents (n)"). "Connectors and quality" is a real link to `/app/connectors`, marked current there. There is no Quality item, no Insights group and no quality state in the sidebar. Collapsed to icons, every item keeps its name as `sr-only` text, and the collapse button is "Collapse sidebar" or "Expand sidebar" with `aria-expanded`. Below `md` a button named "Open menu" opens the sidebar as a drawer, with base-ui's focus trap and Escape. *Checked by* `shell/Sidebar.test.tsx`, `shell/SidebarDrawer.test.tsx` and `shell/AppShell.test.tsx`.
- **The Documents page:** the file input, the Search box and the Type select each have a visible label, each delete button is named `Delete <title>`, deleting asks for confirmation in an alert dialog, and load/delete failures appear in a `role="status"` region. *Checked by* `documents/DocumentList.test.tsx` and `documents/DocumentsPage.test.tsx`.
- **The Connectors page:** each connector is a card (`<section>` named by the connector). Passes, Below bar and Not measured are words next to an `aria-hidden` icon, never colour alone; the metric bars are `aria-hidden` and the number is text; a configuration mismatch ("This result is not current") and load or error messages are `role="status"`. The two external links ("How it's measured", "Connector guide") are real `<a>` elements with a visible focus ring and `rel="noreferrer"`, and the "Build your own" card appears only once the page has loaded. *Checked by* `connectors/ConnectorCard.test.tsx`, `connectors/ConnectorsPage.test.tsx`, `shell/Sidebar.test.tsx` and `shell/a11y.test.tsx`.
- **The first-run screen:** one `<h1>`; Upload documents is a real link to `/app/documents` (plain click only is handled, as in the sidebar); Try the sample documents is a native button, disabled while it runs; a failure appears in a `role="status"` line with a generic message. *Checked by* `chat/FirstRun.test.tsx` and `shell/a11y.test.tsx`.
- **Feedback controls:** "Good answer" and "Bad answer" are named buttons
  with `aria-pressed`; the thumbs-down form is a native `<fieldset>` with a
  `<legend>`, radio inputs and a labelled textarea; the outcome
  ("Feedback recorded" or a generic failure) is a `role="status"` line.
  *Checked by* `feedback/FeedbackControls.test.tsx` and
  `components/assistant-ui/elements/thread.aui.test.tsx`.
- **Async results are announced:** the upload outcome on the Documents
  page lives in a `role="status"` / `aria-live="polite"` region.
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

`useRoute` renders as `chat` on the server, so page-specific markup is
tested through `AppShell` with an explicit `page`. A component with no
hooks, like `Sidebar`, can be called as a function; walk the element it
returns and call a handler with a fake event (`shell/SidebarDrawer.test.tsx`).
Stub `window` with `vi.stubGlobal` for code that uses the History API
(`shell/route.test.ts`). `AppShell` takes the saved chat's load result as a
prop (`saved`), so its loading, not-found and error screens are static tests too.
What a static render cannot show: that a restored thread keeps its
`metadata.custom` after the runtime loads `initialMessages`, the sidebar list
refreshing after a rename or delete, focus moving into the rename input, and
the thread's height under the chat header. Check those by hand in the running
app.

A read-only thread can't run or cancel, so Stop, Copy (the clipboard),
Regenerate and scrolling are checked by hand in a browser, and so is the
feedback form (Tab order through Copy, Regenerate, Good, Bad; Enter and
Space; arrow keys across the reasons; Cancel and Send return focus to the
Bad button; focus rings; light and dark themes) and the
Documents page (Tab order, search and type filter, upload outcome, delete
confirm and cancel, focus rings, light and dark themes) and the
Connectors page (Tab reaches the Connectors and quality link, Enter opens
the page, focus rings, light and dark themes) and the first-run screen (Tab
reaches Upload documents then Try the sample documents, focus rings, light
and dark themes; the samples button indexes them, the screen gives way to the
chat with the sample question, and deleting every document brings it back)
and the shell: New
chat clears a conversation, including while an answer streams; going to
Documents and back leaves the chat unchanged; the drawer opens from the menu
button, closes on Escape, the backdrop and navigation, and returns focus to
the menu button; the collapsed state survives a reload; resizing across
768 px switches between the inline sidebar and the menu button.

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
