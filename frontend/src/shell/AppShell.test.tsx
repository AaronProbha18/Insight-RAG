import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Page } from './route'
import { AppShell } from './AppShell'

const doc = { document_id: 'doc_1', title: 'a.md', connector: 'markdown', created_at: '2026-01-01' }
const documents = { documents: [doc], loaded: true, message: '', refresh: () => {}, remove: async () => {} }
const render = (page: Page, docs: Partial<typeof documents> = {}) =>
  renderToStaticMarkup(
    <AppShell
      page={page}
      collapsed={false}
      onToggleCollapse={() => {}}
      chatKey={0}
      onNewChat={() => {}}
      documents={{ ...documents, ...docs }}
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

  describe('first run', () => {
    const HEADLINE = 'Ask questions about your documents'
    const empty = { documents: [] }

    it('shows first-run, not the chat, with no documents, and the sidebar counts zero', () => {
      const html = render('chat', empty)
      expect(html).toContain(HEADLINE)
      expect(html).toContain('Documents (0)')
      expect(html).toContain('<div hidden=""')
    })

    it('shows nothing for the chat page until the list has loaded', () => {
      const html = render('chat', { ...empty, loaded: false })
      expect(html).not.toContain(HEADLINE)
      expect(html).not.toContain(THREAD)
    })

    it('shows the chat, not first-run, when documents exist; first-run again once all are gone', () => {
      expect(render('chat')).not.toContain(HEADLINE)
      expect(render('chat', empty)).toContain(HEADLINE)
    })

    it('shows the chat when the list failed to load', () => {
      const html = render('chat', { ...empty, message: "Couldn't load documents. Try again." })
      expect(html).not.toContain(HEADLINE)
      expect(html).toContain(THREAD)
    })

    it('does not show first-run on the Documents page', () => {
      expect(render('documents', empty)).not.toContain(HEADLINE)
    })
  })
})
