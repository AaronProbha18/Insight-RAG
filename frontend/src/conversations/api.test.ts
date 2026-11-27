import { afterEach, describe, expect, it, vi } from 'vitest'
import { deleteConversation, getConversation, listConversations, renameConversation } from './api'

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

const list = [
  { conversation_id: 'conv_B', title: 'Newer', updated_at: '2026-11-02T10:00:00Z' },
  { conversation_id: 'conv_A', title: 'Older', updated_at: '2026-11-01T10:00:00Z' },
]

const detail = {
  conversation_id: 'conv_A',
  title: 'Refunds',
  turns: [
    {
      answer_id: 'ans_1',
      question: 'Refund window?',
      answer: '14 days [c1]',
      abstained: false,
      citations: [{ document_id: 'doc_1', title: 'Returns', locator: 'p. 1', indexed: false }],
    },
  ],
}

describe('listConversations', () => {
  it('returns the conversations in the order given', async () => {
    respond(200, list)

    expect(await listConversations()).toEqual(list)
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/conversations')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('throws on a server error and when offline', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })
    await expect(listConversations()).rejects.toThrow()
    offline()
    await expect(listConversations()).rejects.toThrow()
  })
})

describe('getConversation', () => {
  it('returns the turns with their citations', async () => {
    respond(200, detail)

    expect(await getConversation('conv_A')).toEqual(detail)
    expect((fetch as any).mock.calls[0][0]).toBe('/conversations/conv_A')
  })

  it('encodes the id in the path', async () => {
    respond(200, detail)

    await getConversation('a/b')

    expect((fetch as any).mock.calls[0][0]).toBe('/conversations/a%2Fb')
  })

  it('returns null on 404', async () => {
    respond(404, { detail: 'Conversation not found' })

    expect(await getConversation('conv_X')).toBeNull()
  })

  it('throws on a server error and when offline', async () => {
    respond(500)
    await expect(getConversation('conv_A')).rejects.toThrow()
    offline()
    await expect(getConversation('conv_A')).rejects.toThrow()
  })
})

describe('renameConversation', () => {
  it('PATCHes the title and returns the stored one', async () => {
    respond(200, { conversation_id: 'conv_A', title: 'Returns', updated_at: 'x' })

    expect(await renameConversation('conv_A', ' Returns ')).toEqual({ kind: 'renamed', title: 'Returns' })
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/conversations/conv_A')
    expect(init.method).toBe('PATCH')
    expect(init.headers['Content-Type']).toBe('application/json')
    expect(JSON.parse(init.body)).toEqual({ title: ' Returns ' })
  })

  it('reports invalid on 422 and not-found on 404', async () => {
    respond(422, { detail: 'title must be 1-200 characters' })
    expect(await renameConversation('conv_A', '')).toEqual({ kind: 'invalid' })
    respond(404, { detail: 'Conversation not found' })
    expect(await renameConversation('conv_A', 'x')).toEqual({ kind: 'not-found' })
  })

  it('throws on a server error and when offline', async () => {
    respond(500)
    await expect(renameConversation('conv_A', 'x')).rejects.toThrow()
    offline()
    await expect(renameConversation('conv_A', 'x')).rejects.toThrow()
  })
})

describe('deleteConversation', () => {
  it('reports deleted on 204', async () => {
    respond(204)

    expect(await deleteConversation('conv_A')).toBe('deleted')
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/conversations/conv_A')
    expect(init.method).toBe('DELETE')
  })

  it('encodes the id and reports not-found on 404', async () => {
    respond(404)

    expect(await deleteConversation('a/b')).toBe('not-found')
    expect((fetch as any).mock.calls[0][0]).toBe('/conversations/a%2Fb')
  })

  it('throws on a server error and when offline', async () => {
    respond(500)
    await expect(deleteConversation('conv_A')).rejects.toThrow()
    offline()
    await expect(deleteConversation('conv_A')).rejects.toThrow()
  })
})
