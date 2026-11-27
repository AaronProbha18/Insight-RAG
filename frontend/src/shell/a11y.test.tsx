/// <reference types="node" /> // the source scan reads files from disk
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { FirstRunView } from '../chat/FirstRun'
import { ConnectorsView } from '../connectors/ConnectorsPage'
import type { Quality } from '../quality/api'
import { AppShell } from './AppShell'
import type { Page } from './route'

const documents = { documents: [], loaded: true, message: '', refresh: () => {}, remove: async () => {} }
const render = (page: Page, collapsed: boolean) =>
  renderToStaticMarkup(
    <AppShell
      page={page}
      collapsed={collapsed}
      onToggleCollapse={() => {}}
      chatKey={0}
      onNewChat={() => {}}
      documents={documents}
      quality={{ quality: null, message: '' }}
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
  ['connectors', false],
  ['connectors', true],
] as const)('AppShell on %s, collapsed=%s', (page, collapsed) => {
  const html = render(page, collapsed)

  it('gives every control an accessible name', () => {
    expect(unnamed(html)).toEqual([])
  })

  it('announces the current page once, on Documents or on Connectors and quality', () => {
    const current = html.match(/aria-current="page"/g) ?? []
    expect(current.length).toBe({ chat: 0, documents: 1, connectors: 1 }[page])
  })
})

describe('shell sources', () => {
  const files = [
    'Sidebar.tsx',
    'SidebarDrawer.tsx',
    'AppShell.tsx',
    '../chat/FirstRun.tsx',
    '../documents/DocumentsPage.tsx',
    '../connectors/ConnectorCard.tsx',
    '../connectors/ConnectorsPage.tsx',
  ]
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

describe('keyboard reach on the Connectors page', () => {
  const html = render('connectors', false)

  it('has only native links: each has an href, none is removed from the tab order or disabled', () => {
    const links = html.match(/<a\b[^>]*>/g) ?? []

    expect(links.length).toBeGreaterThanOrEqual(4) // Documents, Connectors and quality, Docs, How it's measured
    for (const l of links) {
      expect(l).toContain('href="')
      expect(l).not.toMatch(/tabindex="-1"|aria-disabled/)
    }
    expect(html).not.toContain('aria-disabled')
  })
})

describe('the loaded Connectors page', () => {
  const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
  const quality: Quality = {
    state: 'measured',
    result: 'FAIL',
    run_at: '2026-11-25T09:00:00+00:00',
    models: { chat: 'c', embeddings: 'e', judge: 'j' },
    threshold: 0.3,
    config_matches: false,
    config_differences: ['chat'],
    connectors: {
      pdf: { result: 'PASS', failed: [], metrics },
      docx: { result: 'FAIL', failed: ['correctness'], metrics: { ...metrics, correctness: 0.7 } },
    },
  }
  const connector = (name: string) => ({ name, extensions: ['.' + name], contract_version: 1, builtin: true })
  const html = renderToStaticMarkup(
    <ConnectorsView
      connectors={['docx', 'pdf', 'xlsx'].map(connector)}
      connectorsMessage=""
      quality={quality}
      qualityMessage=""
      documents={[]}
    />,
  )

  it('names every control, and each card is a labelled section', () => {
    expect(unnamed(html)).toEqual([])
    expect(html.match(/<section aria-label="[^"]+"/g)).toHaveLength(4) // three cards and Build your own
  })

  it('has headings in order: one h1, then h2s', () => {
    const levels = [...html.matchAll(/<h([1-6])\b/g)].map((m) => Number(m[1]))

    expect(levels[0]).toBe(1)
    expect(levels.filter((l) => l === 1)).toHaveLength(1)
    expect(levels.slice(1).every((l) => l === 2)).toBe(true)
  })

  it('shows Passes, Below bar and Not measured as a word beside an aria-hidden icon', () => {
    for (const word of ['Passes', 'Below bar', 'Not measured'])
      expect(html).toMatch(new RegExp('<svg[^>]*aria-hidden="true"[^>]*>(?:(?!</span>).)*</svg>' + word))
  })

  it('has external links with an href, rel and visible text, and nothing off the tab order', () => {
    const external = html.match(/<a [^>]*target="_blank"[^>]*>[^<]+<\/a>/g) ?? []

    expect(external).toHaveLength(2)
    for (const a of external) {
      expect(a).toContain('href="https://github.com/')
      expect(a).toContain('rel="noreferrer"')
    }
    expect(html).not.toMatch(/tabindex|aria-disabled/)
  })
})

describe('Connectors page sources', () => {
  const source = (f: string) => readFileSync(fileURLToPath(new URL(f, import.meta.url)), 'utf-8')
  // Every text/background pair these colours form is in PAIRS in design-tokens.test.ts.
  const MEASURED = ['text-foreground', 'text-muted-foreground', 'text-destructive', 'border-border', 'border-destructive', 'bg-foreground', 'bg-muted', 'outline-ring']
  const COLOUR = /\b(?:text|bg|border|outline)-(?:muted-foreground|primary-foreground|secondary-foreground|accent-foreground|popover-foreground|foreground|background|muted|destructive|border|primary|secondary|accent|popover|card|ring)(?![A-Za-z-])/g

  it.each(['../connectors/ConnectorCard.tsx', '../connectors/ConnectorsPage.tsx'])(
    '%s uses only colours whose contrast is measured',
    (f) => {
      for (const c of source(f).match(COLOUR) ?? []) expect(MEASURED, c).toContain(c)
    },
  )

  it('gives the page links a visible focus style', () => {
    expect(source('../connectors/ConnectorsPage.tsx')).toMatch(/const link =\s+'[^']*focus-visible:outline-2/)
  })
})

describe('the first-run screen', () => {
  const html = renderToStaticMarkup(<FirstRunView error="" busy={false} onUpload={() => {}} onTry={() => {}} />)
  const source = readFileSync(fileURLToPath(new URL('../chat/FirstRun.tsx', import.meta.url)), 'utf-8')
  const MEASURED = ['text-muted-foreground', 'text-foreground', 'text-destructive', 'bg-primary', 'text-primary-foreground', 'bg-muted', 'border-border']
  const COLOUR = /\b(?:text|bg|border|outline)-(?:muted-foreground|primary-foreground|foreground|background|muted|destructive|border|primary|ring)(?![A-Za-z-])/g

  it('names every control and has one h1', () => {
    expect(unnamed(html)).toEqual([])
    expect([...html.matchAll(/<h([1-6])\b/g)].map((m) => m[1])).toEqual(['1'])
  })

  it('has a native link and a native button, both in the tab order, with a focus style', () => {
    const link = html.match(/<a\b[^>]*>/)![0]
    const button = html.match(/<button\b[^>]*>/)![0]

    expect(link).toContain('href="/app/documents"')
    expect(link).toContain('focus-visible:')
    expect(button).toContain('focus-visible:')
    for (const el of [link, button]) expect(el).not.toMatch(/tabindex="-1"|aria-disabled|\sdisabled=/)
  })

  it('announces an error in a status region', () => {
    expect(html).toContain('role="status"')
  })

  it('uses only colours whose contrast is measured in design-tokens.test.ts', () => {
    for (const c of source.match(COLOUR) ?? []) expect(MEASURED, c).toContain(c)
  })
})
