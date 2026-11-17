import { afterEach, describe, expect, it, vi } from 'vitest'
import { uploadDocument } from './api'

const file = new File(['%PDF'], 'returns.pdf', { type: 'application/pdf' })

function respond(status: number, body: object) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(body), { status })))
}

afterEach(() => vi.unstubAllGlobals())

describe('uploadDocument', () => {
  it('reports a newly indexed document', async () => {
    respond(201, { title: 'Returns Policy', chunks: 3, created: true })

    expect(await uploadDocument(file)).toEqual({
      kind: 'indexed',
      message: 'Indexed "Returns Policy" (3 sections).',
    })
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/documents')
    expect(init.method).toBe('POST')
    expect((init.body as FormData).get('file')).toBeInstanceOf(File)
  })

  it('reports an already indexed document', async () => {
    respond(200, { title: 'Returns Policy', chunks: 3, created: false })

    expect(await uploadDocument(file)).toEqual({
      kind: 'already-indexed',
      message: '"Returns Policy" is already indexed.',
    })
  })

  it('shows the supported types for an unsupported file', async () => {
    respond(415, { detail: "unsupported file type '.png'; supported: .docx, .pdf" })

    const outcome = await uploadDocument(file)
    expect(outcome.kind).toBe('unsupported')
    expect(outcome.message).toContain('.docx, .pdf')
  })

  it('reports an unreadable file', async () => {
    respond(422, { detail: 'unreadable PDF: Stream has ended unexpectedly' })

    expect((await uploadDocument(file)).kind).toBe('unreadable')
  })

  it('hides internals on server errors', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })

    expect(await uploadDocument(file)).toEqual({ kind: 'error', message: 'Upload failed. Try again.' })
  })

  it('handles a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))

    expect((await uploadDocument(file)).kind).toBe('error')
  })
})
