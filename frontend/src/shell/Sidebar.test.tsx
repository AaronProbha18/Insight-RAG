/// <reference types="node" />
import { existsSync } from 'node:fs'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Page } from './route'
import { Sidebar } from './Sidebar'

const render = (page: Page = 'chat', collapsed = false) =>
  renderToStaticMarkup(
    <Sidebar
      page={page}
      documentCount={3}
      collapsed={collapsed}
      onToggleCollapse={() => {}}
      onNewChat={() => {}}
    />,
  )
const links = (html: string, href: string) => html.match(new RegExp(`<a[^>]*href="${href}"[^>]*>`, 'g')) ?? []

describe('Sidebar', () => {
  it('lists logo, New chat, Knowledge and Docs in order', () => {
    const html = render()
    const order = ['Plumbline', 'New chat', 'Knowledge', 'Documents (3)', 'Connectors and quality', 'Docs<']
    const at = order.map((t) => html.indexOf(t))

    expect(at.every((i) => i >= 0)).toBe(true)
    expect(at).toEqual([...at].sort((a, b) => a - b))
    expect(html).toContain('aria-label="Main"')
    expect(html).not.toContain('Coming soon')
    expect(html).not.toContain('Insights')
    expect(html).not.toContain('>Quality<')
  })

  it('marks Documents as the current page only on that page', () => {
    const link = (html: string) => html.match(/<a[^>]*href="\/app\/documents"[^>]*>/)?.[0] ?? ''

    expect(link(render('documents'))).toContain('aria-current="page"')
    expect(link(render('chat'))).not.toContain('aria-current')
  })

  it('makes Connectors and quality the one link to the page, current only there', () => {
    expect(links(render(), '/app/connectors')).toHaveLength(1)
    expect(render()).not.toContain('aria-disabled')
    expect(links(render('connectors'), '/app/connectors')[0]).toContain('aria-current="page"')
    expect(links(render('chat'), '/app/connectors')[0]).not.toContain('aria-current')
  })

  it('has no quality state in the sidebar', () => {
    expect(render()).not.toMatch(/Quality: |Passes|Below bar|Not measured/)
  })

  it('has no quality dialog any more', () => {
    expect(render()).not.toMatch(/aria-haspopup|data-slot="dialog/)
    for (const f of ['Quality.tsx', 'QualityPanel.tsx', 'label.ts'])
      expect(existsSync(new URL(`../quality/${f}`, import.meta.url))).toBe(false)
  })

  it('collapsed keeps every name as screen-reader text and offers Expand', () => {
    const html = render('chat', true)

    expect(html).toContain('aria-label="Expand sidebar"')
    expect(html).toContain('aria-expanded="false"')
    for (const name of ['Plumbline', 'New chat', 'Documents (3)', 'Connectors and quality', 'Docs'])
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

    expect(triggers(render('chat', true))).toBe(5) // expand, New chat, Documents, Connectors and quality, Docs
    expect(triggers(render('chat', false))).toBe(0)
  })

  it('expanded offers Collapse', () => {
    expect(render()).toContain('aria-label="Collapse sidebar"')
  })
})
