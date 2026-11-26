import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Page } from './route'
import { AppShell } from './AppShell'

const documents = { documents: [], message: '', refresh: () => {}, remove: async () => {} }
const render = (page: Page) =>
  renderToStaticMarkup(
    <AppShell
      page={page}
      collapsed={false}
      onToggleCollapse={() => {}}
      chatKey={0}
      onNewChat={() => {}}
      documents={documents}
      quality={{ quality: null, message: '' }}
    />,
  )
const THREAD = 'aria-label="Ask a question about your documents"'

describe('AppShell', () => {
  it('shows the chat, not the Documents page, on the chat page', () => {
    const html = render('chat')

    expect(html).toContain(THREAD)
    expect(html).not.toContain('<div hidden=""')
    expect(html).not.toContain('<h1')
  })

  it('keeps the chat mounted but hidden on the Documents page', () => {
    const html = render('documents')

    expect(html).toContain('<h1 class="mb-4 text-xl font-semibold">Documents</h1>')
    const hidden = html.indexOf('<div hidden=""')
    expect(hidden).toBeGreaterThanOrEqual(0)
    expect(html.indexOf(THREAD)).toBeGreaterThan(hidden)
    expect(html.indexOf(THREAD)).toBeLessThan(html.indexOf('<h1'))
  })

  it('shows the Connectors page, with the chat mounted but hidden, on the connectors page only', () => {
    const html = render('connectors')

    expect(html).toContain('>Connectors and quality</h1>')
    expect(html.indexOf('<div hidden=""')).toBeGreaterThanOrEqual(0)
    expect(html.indexOf(THREAD)).toBeLessThan(html.indexOf('<h1'))
    for (const page of ['chat', 'documents'] as const) expect(render(page)).not.toContain('>Connectors and quality</h1>')
  })

  it('hides the inline sidebar below md and offers a menu button there', () => {
    const html = render('chat')

    expect(html).toContain('class="hidden md:block"')
    expect(html).toContain('aria-label="Open menu"')
  })

  it('has no upload bar on any page but Documents', () => {
    expect(render('chat')).not.toContain('Upload a PDF or Word document')
    expect(render('documents')).toContain('Upload a PDF or Word document')
  })
})
