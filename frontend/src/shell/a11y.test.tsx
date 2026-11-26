/// <reference types="node" /> // the source scan reads files from disk
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AppShell } from './AppShell'
import type { Page } from './route'

const documents = { documents: [], message: '', refresh: () => {}, remove: async () => {} }
const render = (page: Page, collapsed: boolean) =>
  renderToStaticMarkup(
    <AppShell
      page={page}
      collapsed={collapsed}
      onToggleCollapse={() => {}}
      chatKey={0}
      onNewChat={() => {}}
      documents={documents}
      quality={{ quality: null, message: '', refresh: () => {} }}
    />,
  )

const text = (html: string) => html.replace(/<[^>]*>/g, '').trim()

/** Controls without an accessible name: aria-label, own text (sr-only counts), or a wrapping <label>. */
function unnamed(html: string): string[] {
  const bad: string[] = []
  for (const m of html.matchAll(/<(button|a)\b([^>]*)>([\s\S]*?)<\/\1>/g))
    if (!/aria-label="[^"]+"/.test(m[2]) && !text(m[3])) bad.push(m[0].slice(0, 80))
  const labelled = [...html.matchAll(/<label\b[^>]*>([\s\S]*?)<\/label>/g)].map((m) => m[1])
  for (const m of html.matchAll(/<(input|select|textarea)\b([^>]*)>/g))
    if (!/aria-label="[^"]+"/.test(m[2]) && !labelled.some((l) => l.includes(m[0]) && text(l))) bad.push(m[0].slice(0, 80))
  return bad
}

describe.each([
  ['chat', false],
  ['chat', true],
  ['documents', false],
  ['documents', true],
] as const)('AppShell on %s, collapsed=%s', (page, collapsed) => {
  const html = render(page, collapsed)

  it('gives every control an accessible name', () => {
    expect(unnamed(html)).toEqual([])
  })

  it('announces the current page once, on Documents only', () => {
    const current = html.match(/aria-current="page"/g) ?? []
    expect(current.length).toBe(page === 'documents' ? 1 : 0)
  })
})

describe('shell sources', () => {
  const files = ['Sidebar.tsx', 'SidebarDrawer.tsx', 'AppShell.tsx', '../documents/DocumentsPage.tsx']
  const source = (f: string) => readFileSync(fileURLToPath(new URL(f, import.meta.url)), 'utf-8')

  it.each(files)('%s never removes the outline without a focus-visible replacement', (f) => {
    for (const m of source(f).matchAll(/(['"`])((?:(?!\1).)*outline-none(?:(?!\1).)*)\1/g))
      expect(m[2], m[2]).toContain('focus-visible:')
  })

  it.each(files)('%s uses tokens, never colour literals or palette classes', (f) => {
    expect(source(f)).not.toMatch(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(/)
    expect(source(f)).not.toMatch(/\b(text|bg|border|ring)-(red|amber|blue|green|gray|slate|zinc|neutral|stone|orange|yellow)-\d/)
  })
})
