import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { UploadOutcome } from '../upload/api'
import type { DocumentSummary } from './api'
import { DocumentsPage } from './DocumentsPage'
import { filterDocuments, uploadAndRefresh } from './pageLogic'

afterEach(() => vi.unstubAllGlobals())

const alpha: DocumentSummary = { document_id: 'doc_1', title: 'Alpha', connector: 'pdf', created_at: '2026-01-02T00:00:00Z' }
const beta: DocumentSummary = { document_id: 'doc_2', title: 'Beta', connector: 'docx', created_at: '2026-01-01T00:00:00Z' }
const noop = async () => {}

describe('filterDocuments', () => {
  it('searches the title in any case', () => {
    expect(filterDocuments([alpha, beta], 'ALP', 'all')).toEqual([alpha])
  })

  it('filters by type', () => {
    expect(filterDocuments([alpha, beta], '', 'docx')).toEqual([beta])
    expect(filterDocuments([alpha, beta], '', 'pdf')).toEqual([alpha])
    expect(filterDocuments([alpha, beta], 'beta', 'pdf')).toEqual([])
  })

  it('keeps the given order with no query and all types', () => {
    expect(filterDocuments([alpha, beta], '', 'all')).toEqual([alpha, beta])
  })
})

describe('DocumentsPage', () => {
  const render = (documents: DocumentSummary[]) =>
    renderToStaticMarkup(<DocumentsPage documents={documents} message="" refresh={() => {}} remove={noop} />)

  it('lists documents newest first with type and delete buttons', () => {
    const html = render([alpha, beta])

    expect(html.indexOf('Alpha')).toBeLessThan(html.indexOf('Beta'))
    expect(html).toContain('PDF')
    expect(html).toContain('DOCX')
    expect(html).toContain('aria-label="Delete Alpha"')
  })

  it('shows the empty state that points to upload', () => {
    expect(render([])).toContain('No documents yet. Upload a PDF or DOCX to get started.')
  })

  it('names every control and has a status region', () => {
    const html = render([alpha])

    expect(html).toMatch(/<label[^>]*>Upload a PDF or Word document\s*<input type="file" accept="\.pdf,\.docx"/)
    expect(html).toMatch(/Search\s*<input[^>]*type="search"/)
    expect(html).toMatch(/Type\s*<select/)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-live="polite"')
  })
})

describe('uploadAndRefresh', () => {
  const file = new File(['x'], 'a.pdf')
  const outcome = (kind: UploadOutcome['kind']): UploadOutcome => ({ kind, message: kind })

  it('refreshes once for a newly indexed file and returns the outcome', async () => {
    const refresh = vi.fn()
    expect(await uploadAndRefresh(file, refresh, async () => outcome('indexed'))).toEqual(outcome('indexed'))
    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it.each(['already-indexed', 'unsupported', 'unreadable', 'error'] as const)('does not refresh for %s', async (kind) => {
    const refresh = vi.fn()
    await uploadAndRefresh(file, refresh, async () => outcome(kind))
    expect(refresh).not.toHaveBeenCalled()
  })

  it('refreshes after a real 201 through fetch', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Response.json({ title: 'A', chunks: 3 }, { status: 201 })))
    const refresh = vi.fn()
    const result = await uploadAndRefresh(file, refresh)
    expect(result.kind).toBe('indexed')
    expect(refresh).toHaveBeenCalledTimes(1)
  })
})
