import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Page } from './route'
import { Sidebar } from './Sidebar'

const render = (page: Page = 'chat', collapsed = false) =>
  renderToStaticMarkup(
    <Sidebar page={page} documentCount={3} collapsed={collapsed} onToggleCollapse={() => {}} onNewChat={() => {}} />,
  )

describe('Sidebar', () => {
  it('lists logo, New chat, Knowledge, Insights and Docs in order', () => {
    const html = render()
    const order = ['Plumbline', 'New chat', 'Knowledge', 'Documents (3)', 'Connectors', 'Insights', 'Quality', 'Docs<']
    const at = order.map((t) => html.indexOf(t))

    expect(at.every((i) => i >= 0)).toBe(true)
    expect(at).toEqual([...at].sort((a, b) => a - b))
    expect(html).toContain('aria-label="Main"')
    expect(html).toContain('Coming soon')
  })

  it('marks Documents as the current page only on that page', () => {
    const link = (html: string) => html.match(/<a[^>]*href="\/app\/documents"[^>]*>/)?.[0] ?? ''

    expect(link(render('documents'))).toContain('aria-current="page"')
    expect(link(render('chat'))).not.toContain('aria-current')
  })

  it('does not make Connectors a link', () => {
    const html = render()

    expect(html).toMatch(/<span[^>]*aria-disabled="true"/)
    expect(html.match(/<a[^>]*>/g)?.length).toBe(2) // Documents and Docs only
  })

  it('collapsed keeps every name as screen-reader text and offers Expand', () => {
    const html = render('chat', true)

    expect(html).toContain('aria-label="Expand sidebar"')
    expect(html).toContain('aria-expanded="false"')
    for (const name of ['Plumbline', 'New chat', 'Documents (3)', 'Docs'])
      expect(html).toContain(`<span class="sr-only">${name}</span>`)
  })

  it('expanded offers Collapse', () => {
    expect(render()).toContain('aria-label="Collapse sidebar"')
  })
})
