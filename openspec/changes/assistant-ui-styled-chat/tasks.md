## 1. Spike and decision record

- [x] 1.1 Spike on a throwaway branch: run the assistant-ui CLI against `frontend/` and record in `design.md` the exact install command, Tailwind version, `components.json`/`@/` alias needs, added packages, dark-mode mechanism, markdown/GFM support, and whether kit components render under `renderToStaticMarkup`; resolve every UNVERIFIED in D2-D7. No spike code is merged.
- [x] 1.2 Write `docs/adr/0003-styled-chat-kit.md` (Proposed) from D1-D3 and the spike findings; link it from `design.md`.

## 2. Styling foundation

- [ ] 2.1 Add the Tailwind + shadcn setup per the spike (deps, config, `@/` alias in `tsconfig.app.json` and `vite.config.ts` if required); `index.css` becomes the Tailwind entry with shadcn default tokens and the dark block bound to `prefers-color-scheme`. The existing UI still builds; `npm run lint` and `npm run build` pass.
- [ ] 2.2 Rewrite `design-tokens.test.ts` for the new token format: no colour literals outside the token blocks, >= 4.5:1 for every text/background pair used, both themes. Fix any failing default token, with the measured ratio in a comment.

## 3. Chat thread

- [ ] 3.1 Vendor the kit thread components from the saved `registry/*.json` items into `frontend/src/components/` and trim `thread.aui.tsx` per D1, replace the primitive markup in `App.tsx`, and delete the superseded `index.css` rules; test: static markup shows the composer with an accessible name and the Send control.
- [ ] 3.2 Empty state: welcome text and synthetic, corpus-agnostic suggested questions; test: empty-thread markup contains the welcome and >= 1 suggestion with an accessible name.
- [ ] 3.3 Markdown answers through the kit's markdown component (add `remark-gfm` if the spike says it is missing); test: list, code block and table render as elements, and `<script>` in text creates no element.
- [ ] 3.4 Render `Citations` inside the generated assistant message from `metadata.custom.citations`, restyled with the kit; update `Citations.test.tsx`: sources render with title and locator in `<details>`, an empty list renders nothing.
- [ ] 3.5 Action bar and running state: Copy and Regenerate on assistant messages, Stop while running, a scroll-to-bottom control; test: static markup exposes each control with an accessible name.

## 4. Upload

- [ ] 4.1 Restyle `Upload.tsx` with the kit, keeping the labelled `<input type="file">` and the `role="status"` region; update `Upload.test.tsx`.

## 5. Docs

- [ ] 5.1 Update `docs/guides/frontend-development-guide.md` (§1 layout, §4.1 kit not primitives, §6 Tailwind + tokens, §7 test approach, §9 table), the `REVIEW.md` frontend checklist, and the frontend mentions in `docs/architecture/overview.md`.

## 6. Manual check

- [ ] 6.1 Manual pass at 375px and desktop, light and dark OS theme, keyboard only: upload, suggestion, ask, stop, copy, regenerate, scroll-to-bottom, open a source; record the results and the before/after `npm run build` bundle size for `verification.md`.
