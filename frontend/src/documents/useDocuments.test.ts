import { afterEach, describe, expect, it, vi } from 'vitest'
import { deleteOutcome } from './useDocuments'

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
