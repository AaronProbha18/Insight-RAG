import { afterEach, describe, expect, it, vi } from 'vitest'
import { deleteDocument, indexSamples, listDocuments } from './api'

function respond(status: number, body?: object) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })),
  )
}

function offline() {
  vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))
}

afterEach(() => vi.unstubAllGlobals())

const docs = [
  { document_id: 'doc_B', title: 'Handbook', connector: 'docx', created_at: '2026-11-02T10:00:00Z' },
  { document_id: 'doc_A', title: 'Returns', connector: 'pdf', created_at: '2026-11-01T10:00:00Z' },
]

describe('listDocuments', () => {
  it('returns the documents in the order given', async () => {
    respond(200, docs)

    expect(await listDocuments()).toEqual(docs)
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/documents')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('returns an empty list', async () => {
    respond(200, [])

    expect(await listDocuments()).toEqual([])
  })

  it('throws on a server error', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })

    await expect(listDocuments()).rejects.toThrow()
  })

  it('throws on a network failure', async () => {
    offline()

    await expect(listDocuments()).rejects.toThrow()
  })
})

describe('deleteDocument', () => {
  it('reports deleted on 204', async () => {
    respond(204)

    expect(await deleteDocument('doc_A')).toBe('deleted')
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/documents/doc_A')
    expect(init.method).toBe('DELETE')
  })

  it('encodes the id in the path', async () => {
    respond(204)

    await deleteDocument('a/b')

    expect((fetch as any).mock.calls[0][0]).toBe('/documents/a%2Fb')
  })

  it('reports not-found on 404', async () => {
    respond(404, { detail: 'Document not found' })

    expect(await deleteDocument('doc_A')).toBe('not-found')
  })

  it('throws on a server error', async () => {
    respond(500, { detail: 'boom' })

    await expect(deleteDocument('doc_A')).rejects.toThrow()
  })

  it('throws on a network failure', async () => {
    offline()

    await expect(deleteDocument('doc_A')).rejects.toThrow()
  })
})

describe('indexSamples', () => {
  it('POSTs to /documents/samples and returns the parsed result', async () => {
    const result = { indexed: [{ filename: 'returns-policy.md' }], failed: [] }
    respond(200, result)

    expect(await indexSamples()).toEqual(result)
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/documents/samples')
    expect(init.method).toBe('POST')
  })

  it('throws on 404 and 500', async () => {
    respond(404, { detail: 'samples directory not found: /x' })
    await expect(indexSamples()).rejects.toThrow()
    respond(500)
    await expect(indexSamples()).rejects.toThrow()
  })

  it('throws when offline', async () => {
    offline()
    await expect(indexSamples()).rejects.toThrow()
  })
})
