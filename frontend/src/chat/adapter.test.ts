import { afterEach, describe, expect, it, vi } from 'vitest'
import { type ConversationRef, createChatAdapter } from './adapter'

function respond(sse: string, status = 200) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(sse, { status })))
}

async function run(conversation: ConversationRef = {}) {
  const options = {
    messages: [{ role: 'user', content: [{ type: 'text', text: 'refund window?' }] }],
    abortSignal: new AbortController().signal,
  } as any
  const updates = []
  for await (const u of createChatAdapter(conversation).run(options) as AsyncGenerator<any>) updates.push(u)
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
    const gen = createChatAdapter().run({
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

describe('conversation id', () => {
  const DONE =
    'event: abstain\ndata: {"message": "No."}\n\n' +
    'event: done\ndata: {"answer_id": "ans_1", "conversation_id": "conv_NEW"}\n\n'
  const body = (call = 0) => JSON.parse((fetch as any).mock.calls[call][1].body)

  it('sends no conversation_id for a new chat', async () => {
    respond(DONE)
    await run()
    expect(body()).not.toHaveProperty('conversation_id')
  })

  it('sends the id of a saved chat', async () => {
    respond(DONE)
    await run({ id: 'conv_A' })
    expect(body().conversation_id).toBe('conv_A')
  })

  it('learns the id from done, tells onId once, and sends it on the next question', async () => {
    const onId = vi.fn()
    const conversation: ConversationRef = { onId }
    respond(DONE)

    await run(conversation)
    await run(conversation)

    expect(conversation.id).toBe('conv_NEW')
    expect(onId).toHaveBeenCalledTimes(1)
    expect(onId).toHaveBeenCalledWith('conv_NEW')
    expect(body(1).conversation_id).toBe('conv_NEW')
  })

  it('leaves the chat unsaved when done carries no id', async () => {
    const onId = vi.fn()
    respond('event: abstain\ndata: {"message": "No."}\n\nevent: done\ndata: {"answer_id": "ans_1"}\n\n')
    const conversation: ConversationRef = { onId }

    await run(conversation)

    expect(conversation.id).toBeUndefined()
    expect(onId).not.toHaveBeenCalled()
  })
})
