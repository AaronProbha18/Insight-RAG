import { afterEach, describe, expect, it, vi } from 'vitest'
import { sendFeedback } from './api'

function respond(status: number, body?: object) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('sendFeedback', () => {
  it('reports saved on 204 and posts the wire body', async () => {
    respond(204)

    const result = await sendFeedback({ answerId: 'ans_1', rating: 'down', reason: 'wrong', comment: 'no' })

    expect(result).toBe('saved')
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/feedback')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ answer_id: 'ans_1', rating: 'down', reason: 'wrong', comment: 'no' })
  })

  it('reports not-found on 404', async () => {
    respond(404, { detail: 'Answer not found' })

    expect(await sendFeedback({ answerId: 'ans_x', rating: 'up' })).toBe('not-found')
  })

  it.each([422, 500])('throws on %i without leaking the body', async (status) => {
    respond(status, { detail: 'psycopg OperationalError at db:5432' })

    const error = await sendFeedback({ answerId: 'ans_1', rating: 'up' }).catch((e) => e)

    expect(error).toBeInstanceOf(Error)
    expect(error.message).not.toContain('psycopg')
  })

  it('throws on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))

    await expect(sendFeedback({ answerId: 'ans_1', rating: 'up' })).rejects.toThrow()
  })
})
