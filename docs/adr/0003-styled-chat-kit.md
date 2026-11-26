# ADR 0003: Styled chat kit for the frontend

## Status

Accepted. Introduced by the `assistant-ui-styled-chat` change (issue #3).

## Context

The chat UI is built from assistant-ui primitives (`ThreadPrimitive`,
`MessagePrimitive`, `ComposerPrimitive`) with about 40 lines of plain CSS.
The frontend guide states this as a rule: "primitives, not a styled kit"
(§4.1) and "no CSS framework yet" (§6). The UI has no empty state,
markdown rendering, stop, copy, regenerate or scroll-to-bottom, and each
would be hand-built and hand-styled on the primitives.

assistant-ui publishes a styled kit as shadcn registry items: component
source that is copied into the app, not an npm package. It needs Tailwind
CSS and a shadcn setup. A spike (see the change's `design.md`) confirmed
the requirements and that the kit renders under the existing Node-only
test approach.

## Decision

- The chat UI uses the **assistant-ui styled kit**: shadcn style
  `base-nova` on **Base UI** (`@base-ui/react`), icons from `lucide-react`.
- The kit is **vendored** into `frontend/src/components/` and **trimmed** to
  the features the `/chat` stream can reach. Parts for attachments, tool
  calls, reasoning, voice, branches, message editing and feedback are not
  taken. Vendored files are ours: we edit them directly.
- **Tailwind CSS v4** through `@tailwindcss/vite`. `components.json`
  records the shadcn config so the CLI can add items later.
- **Theme**: shadcn's default `neutral` tokens (`oklch()` values). Dark mode
  follows `prefers-color-scheme`; there is no class-based toggle.
- **Accessibility guarantee is kept**: every text/background pair the UI
  uses meets WCAG AA (>= 4.5:1) in both themes, enforced by
  `design-tokens.test.ts`. A default token that fails is adjusted, with the
  measured ratio in a comment.
- Unchanged: `useLocalRuntime` with our adapter, citations in
  `metadata.custom`, tests with `renderToStaticMarkup` and no jsdom.

## Alternatives

- **Primitives + plain CSS (status quo).** Fewest dependencies, but every
  interaction and its styling is bespoke code to write and maintain.
- **The full, untrimmed kit.** Nine extra registry items for features the
  backend never sends; dead code paths to review and keep accessible.
- **Another chat UI library.** Would replace the assistant-ui runtime and
  our adapter, not just the presentation.

## Consequences

- Supersedes frontend guide §4.1 and §6; the guide, `REVIEW.md` and the
  architecture overview are updated in the same change.
- New dependencies: `tailwindcss`, `@tailwindcss/vite`, `cn`,
  `class-variance-authority`, `@base-ui/react`, `lucide-react`,
  `@assistant-ui/react-markdown`, `remark-gfm`; dev `tw-animate-css`,
  `shadcn`.
- Upstream fixes are not automatic: updating means re-fetching the registry
  items and re-applying the trim.
- The registry (`r.assistant-ui.com`, `ui.shadcn.com`) was unreachable
  from command-line tools on the original dev machine; items can be saved
  from a browser instead. The CLI path still works where the network allows.
- The production bundle grows (Tailwind output, Base UI, icons); the change's
  verification records before and after sizes.
- Amendment (2026-11-24, #13): feedback is no longer excluded. The kit's
  feedback primitives (`ActionBarPrimitive.FeedbackPositive/Negative`
  with a runtime `adapters.feedback`) exist in `@assistant-ui/react`
  0.15.37, but `FeedbackAdapter.submit` returns `void`, carries no reason,
  and cannot report a failed save. The thumbs controls are therefore our
  own component (`feedback/FeedbackControls.tsx`) built from the kit's
  buttons and placed in the assistant action bar, posting to `POST
  /feedback`. The other trimmed kit parts stay excluded.
- Accepted audit finding (2026-11-23): `npm audit` reports 4 moderate
  issues from one chain, `shadcn` -> `ts-morph` -> `@ts-morph/common` ->
  `minimatch` (GHSA-g3hv-75wg-m249). `shadcn` is a dev dependency used only
  for `shadcn/tailwind.css` at build time and for its CLI; nothing in the
  chain ships. `npm audit fix --force` would downgrade `shadcn` to 1.0.0, so
  it is not used. Revisit when a `shadcn` release moves to a fixed
  `ts-morph`.
