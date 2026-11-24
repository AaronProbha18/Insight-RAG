import { afterEach, describe, expect, it, vi } from 'vitest'
import { chatAdapter } from './adapter'

function respond(sse: string, status = 200) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(sse, { status })))
}

async function run() {
  const options = {
    messages: [{ role: 'user', content: [{ type: 'text', text: 'refund window?' }] }],
    abortSignal: new AbortController().signal,
  } as any
  const updates = []
  for await (const u of chatAdapter.run(options) as AsyncGenerator<any>) updates.push(u)
  return updates
}

const CITATION = { id: 'c1', document_id: 'doc_1', title: 'Returns', locator: 'p. 1', text: '14 days' }

afterEach(() => vi.unstubAllGlobals())

describe('chatAdapter', () => {
  it('streams text, then attaches citations', async () => {
    respond(
      'event: delta\ndata: {"text": "14 "}\n\n' +
        'event: delta\ndata: {"text": "days [c1]."}\n\n' +
        `event: citations\ndata: ${JSON.stringify({ citations: [CITATION] })}\n\n` +
        'event: done\ndata: {"answer_id": "ans_1"}\n\n',
    )

    const updates = await run()

    expect(updates.map((u) => u.content[0].text)).toEqual(['14 ', '14 days [c1].', '14 days [c1].', '14 days [c1].'])
    expect(updates.slice(0, -1).every((u) => u.metadata.custom.answerId === undefined)).toBe(true)
    expect(updates.at(-1).metadata.custom).toEqual({ citations: [CITATION], abstained: false, answerId: 'ans_1' })
    const body = JSON.parse((fetch as any).mock.calls[0][1].body)
    expect(body).toEqual({ messages: [{ role: 'user', content: 'refund window?' }] })
  })

  it('marks abstentions and carries no citations', async () => {
    respond('event: abstain\ndata: {"message": "Not in the documents."}\n\nevent: done\ndata: {"answer_id": "ans_2"}\n\n')

    const updates = await run()

    expect(updates[0].content[0].text).toBe('Not in the documents.')
    expect(updates[0].metadata.custom).toEqual({ citations: [], abstained: true })
    expect(updates.at(-1).metadata.custom).toEqual({ citations: [], abstained: true, answerId: 'ans_2' })
  })

  it('has no answerId when done carries none', async () => {
    respond('event: abstain\ndata: {"message": "No."}\n\nevent: done\ndata: {}\n\n')

    expect((await run()).at(-1).metadata.custom.answerId).toBeUndefined()
  })

  it('throws on an error event', async () => {
    respond('event: error\ndata: {"message": "answer generation failed"}\n\n')
    await expect(run()).rejects.toThrow('answer generation failed')
  })

  it('yields no answerId before an error', async () => {
    respond('event: delta\ndata: {"text": "x"}\n\nevent: error\ndata: {"message": "boom"}\n\n')
    const seen: any[] = []
    const gen = chatAdapter.run({
      messages: [{ role: 'user', content: [{ type: 'text', text: 'q' }] }],
      abortSignal: new AbortController().signal,
    } as any) as AsyncGenerator<any>

    await expect(
      (async () => {
        for await (const u of gen) seen.push(u)
      })(),
    ).rejects.toThrow('boom')
    expect(seen.length).toBeGreaterThan(0)
    expect(seen.every((u) => u.metadata.custom.answerId === undefined)).toBe(true)
  })

  it('throws on a non-200 response', async () => {
    respond('{"detail": "bad"}', 422)
    await expect(run()).rejects.toThrow('422')
  })
})
