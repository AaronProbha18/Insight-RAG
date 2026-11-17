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
        'event: done\ndata: {}\n\n',
    )

    const updates = await run()

    expect(updates.map((u) => u.content[0].text)).toEqual(['14 ', '14 days [c1].', '14 days [c1].'])
    expect(updates.at(-1).metadata.custom).toEqual({ citations: [CITATION], abstained: false })
    const body = JSON.parse((fetch as any).mock.calls[0][1].body)
    expect(body).toEqual({ messages: [{ role: 'user', content: 'refund window?' }] })
  })

  it('marks abstentions and carries no citations', async () => {
    respond('event: abstain\ndata: {"message": "Not in the documents."}\n\nevent: done\ndata: {}\n\n')

    const [update] = await run()

    expect(update.content[0].text).toBe('Not in the documents.')
    expect(update.metadata.custom).toEqual({ citations: [], abstained: true })
  })

  it('throws on an error event', async () => {
    respond('event: error\ndata: {"message": "answer generation failed"}\n\n')
    await expect(run()).rejects.toThrow('answer generation failed')
  })

  it('throws on a non-200 response', async () => {
    respond('{"detail": "bad"}', 422)
    await expect(run()).rejects.toThrow('422')
  })
})
