import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { DocumentSummary } from './api'
import { DocumentList } from './DocumentList'

const docs: DocumentSummary[] = [
  { document_id: 'doc_B', title: 'Handbook', connector: 'docx', created_at: '2026-11-02T10:00:00Z' },
  { document_id: 'doc_A', title: 'Returns', connector: 'pdf', created_at: '2026-11-01T10:00:00Z' },
]

const render = (documents: DocumentSummary[]) =>
  renderToStaticMarkup(<DocumentList documents={documents} onDelete={() => {}} />)

describe('DocumentList', () => {
  it('shows title, type and date for each document, in the order given', () => {
    const html = render(docs)

    expect(html.indexOf('Handbook')).toBeGreaterThan(-1)
    expect(html.indexOf('Handbook')).toBeLessThan(html.indexOf('Returns'))
    expect(html).toContain('DOCX')
    expect(html).toContain('PDF')
    for (const d of docs) {
      expect(html).toContain(`<time dateTime="${d.created_at}">${new Date(d.created_at).toLocaleDateString()}</time>`)
    }
  })

  it('points to upload when empty', () => {
    const html = render([])

    expect(html).toContain('No documents yet. Upload a PDF or DOCX to get started.')
    expect(html).not.toContain('<li')
  })

  it('gives every delete button an accessible name', () => {
    const html = render(docs)

    expect(html).toContain('aria-label="Delete Handbook"')
    expect(html).toContain('aria-label="Delete Returns"')
  })

  it('escapes titles', () => {
    const html = render([{ ...docs[0], title: '<img src=x onerror=alert(1)>' }])

    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
  })
})
