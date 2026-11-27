import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { deleteOutcome, loadDocuments, trySamples, useDocuments } from './useDocuments'

afterEach(() => vi.unstubAllGlobals())

function stubFetch(status: number) {
  const fetchMock = vi.fn(async () => new Response(null, { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('deleteOutcome', () => {
  it('is empty after a delete and after an already-gone 404', async () => {
    stubFetch(204)
    expect(await deleteOutcome('doc_1')).toBe('')
    stubFetch(404)
    expect(await deleteOutcome('doc_1')).toBe('')
  })

  it('returns the failure message on a 500 or a network error', async () => {
    stubFetch(500)
    expect(await deleteOutcome('doc_1')).toBe("Couldn't delete the document. Try again.")
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    expect(await deleteOutcome('doc_1')).toBe("Couldn't delete the document. Try again.")
  })

  it('calls DELETE with the encoded id', async () => {
    const fetchMock = stubFetch(204)
    await deleteOutcome('doc_a/b')
    expect(fetchMock).toHaveBeenCalledWith('/documents/doc_a%2Fb', { method: 'DELETE' })
  })
})

describe('loadDocuments', () => {
  it('returns the documents and no message on 200', async () => {
    const list = [{ document_id: 'doc_1', title: 'a.md', connector: 'markdown', created_at: '2026-01-01' }]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(list), { status: 200 })))
    expect(await loadDocuments()).toEqual({ documents: list, message: '' })
  })

  it('returns null and the message on a 500 or a network error', async () => {
    const failed = { documents: null, message: "Couldn't load documents. Try again." }
    stubFetch(500)
    expect(await loadDocuments()).toEqual(failed)
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    expect(await loadDocuments()).toEqual(failed)
  })
})

describe('useDocuments', () => {
  it('is not loaded on the first render', () => {
    function Probe() {
      return <span>{String(useDocuments().loaded)}</span>
    }
    expect(renderToStaticMarkup(<Probe />)).toBe('<span>false</span>')
  })
})

function respondJson(status: number, body?: object) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })))
}

const FAILURE = "Couldn't index the sample documents. Try again."

describe('trySamples', () => {
  it('is empty when something was indexed', async () => {
    respondJson(200, { indexed: [{ filename: 'returns-policy.md' }], failed: [] })
    expect(await trySamples()).toBe('')
  })

  it('shows the error when nothing was indexed', async () => {
    respondJson(200, { indexed: [], failed: [{ filename: 'a.pdf', reason: 'bad' }] })
    expect(await trySamples()).toBe(FAILURE)
  })

  it('shows the same error on a 500 or when offline', async () => {
    respondJson(500)
    expect(await trySamples()).toBe(FAILURE)
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    expect(await trySamples()).toBe(FAILURE)
  })
})

