/// <reference types="node" />
import { existsSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Quality } from '../quality/api'
import type { Page } from './route'
import { Sidebar } from './Sidebar'

const render = (page: Page = 'chat', collapsed = false, quality: Quality | null = null) =>
  renderToStaticMarkup(
    <Sidebar
      page={page}
      documentCount={3}
      quality={quality}
      collapsed={collapsed}
      onToggleCollapse={() => {}}
      onNewChat={() => {}}
    />,
  )
const measured = (result: 'PASS' | 'FAIL'): Quality => ({
  state: 'measured',
  result,
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'c', embeddings: 'e', judge: 'j' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [],
  connectors: {},
})
const links = (html: string, href: string) => html.match(new RegExp(`<a[^>]*href="${href}"[^>]*>`, 'g')) ?? []

describe('Sidebar', () => {
  it('lists logo, New chat, Knowledge, Insights and Docs in order', () => {
    const html = render()
    const order = ['Plumbline', 'New chat', 'Knowledge', 'Documents (3)', 'Connectors', 'Insights', 'Quality', 'Docs<']
    const at = order.map((t) => html.indexOf(t))

    expect(at.every((i) => i >= 0)).toBe(true)
    expect(at).toEqual([...at].sort((a, b) => a - b))
    expect(html).toContain('aria-label="Main"')
    expect(html).not.toContain('Coming soon')
  })

  it('marks Documents as the current page only on that page', () => {
    const link = (html: string) => html.match(/<a[^>]*href="\/app\/documents"[^>]*>/)?.[0] ?? ''

    expect(link(render('documents'))).toContain('aria-current="page"')
    expect(link(render('chat'))).not.toContain('aria-current')
  })

  it('makes Connectors a real link, current only on its page', () => {
    expect(links(render(), '/app/connectors')).toHaveLength(2) // Connectors and Quality
    expect(render()).not.toContain('aria-disabled')
    expect(links(render('connectors'), '/app/connectors').every((l) => l.includes('aria-current="page"'))).toBe(true)
    expect(links(render('chat'), '/app/connectors').some((l) => l.includes('aria-current'))).toBe(false)
  })

  it('names the Quality link by the overall state, in words', () => {
    expect(render()).toContain('>Quality<')
    expect(render('chat', false, { state: 'not_measured', eval_command: 'x' })).toContain('Quality: Not measured')
    expect(render('chat', false, measured('PASS'))).toContain('Quality: Passes')
    expect(render('chat', false, measured('FAIL'))).toContain('Quality: Below bar')
    expect(render('chat', false, measured('FAIL'))).not.toContain('FAIL')
    expect(render('chat', true, measured('PASS'))).toContain('<span class="sr-only">Quality: Passes</span>')
  })

  it('has no quality dialog any more', () => {
    expect(render()).not.toMatch(/aria-haspopup|data-slot="dialog/)
    for (const f of ['Quality.tsx', 'QualityPanel.tsx'])
      expect(existsSync(new URL(`../quality/${f}`, import.meta.url))).toBe(false)
  })

  it('collapsed keeps every name as screen-reader text and offers Expand', () => {
    const html = render('chat', true)

    expect(html).toContain('aria-label="Expand sidebar"')
    expect(html).toContain('aria-expanded="false"')
    for (const name of ['Plumbline', 'New chat', 'Documents (3)', 'Connectors', 'Quality', 'Docs'])
      expect(html).toContain(`<span class="sr-only">${name}</span>`)
  })

  it('collapsed keeps the logo and reveals the expand button over it on hover or focus', () => {
    const html = render('chat', true)
    const button = html.match(/<button[^>]*aria-label="Expand sidebar"[^>]*>/)?.[0] ?? ''

    expect(html).toContain('<img src="/favicon.svg" alt="" class="size-5 shrink-0"')
    expect(button).toContain('opacity-0')
    expect(button).toContain('group-hover/head:opacity-100')
    expect(button).toContain('group-has-[:focus-visible]/head:opacity-100') // keyboard focus reveals it; mouse focus after a click must not
    expect(button).not.toContain('hidden')
  })

  it('expanded shows the collapse button with no hover trick', () => {
    expect(render('chat', false).match(/<button[^>]*aria-label="Collapse sidebar"[^>]*>/)?.[0]).not.toContain('opacity-0')
  })

  it('has tooltips on the icon-only items, and only when collapsed', () => {
    const triggers = (html: string) => html.match(/data-slot="tooltip-trigger"/g)?.length ?? 0

    expect(triggers(render('chat', true))).toBe(6) // expand, New chat, Documents, Connectors, Quality, Docs
    expect(triggers(render('chat', false))).toBe(0)
  })

  it('expanded offers Collapse', () => {
    expect(render()).toContain('aria-label="Collapse sidebar"')
  })
})
