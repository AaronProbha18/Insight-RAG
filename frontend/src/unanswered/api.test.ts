import { afterEach, describe, expect, it, vi } from 'vitest'
import { listUnanswered } from './api'

afterEach(() => vi.unstubAllGlobals())

function stubFetch(status: number, body?: object) {
  const fetchMock = vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('listUnanswered', () => {
  it('asks for all time without a days parameter', async () => {
    const fetchMock = stubFetch(200, { items: [] })
    expect(await listUnanswered(null)).toEqual([])
    expect(fetchMock).toHaveBeenCalledWith('/unanswered')
  })

  it.each([7, 30, 90] as const)('asks for the last %i days', async (days) => {
    const fetchMock = stubFetch(200, { items: [] })
    await listUnanswered(days)
    expect(fetchMock).toHaveBeenCalledWith(`/unanswered?days=${days}`)
  })

  it('returns the items', async () => {
    const item = { question: 'Q?', count: 2, last_asked_at: '2026-01-01', reason: 'marked_wrong', feedback_reason: 'wrong' }
    stubFetch(200, { items: [item] })
    expect(await listUnanswered(30)).toEqual([item])
  })

  it('throws on a 500', async () => {
    stubFetch(500)
    await expect(listUnanswered(30)).rejects.toThrow()
  })
})
