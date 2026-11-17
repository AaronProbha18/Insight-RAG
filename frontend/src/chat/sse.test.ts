import { describe, expect, it } from 'vitest'
import { parseSse } from './sse'

/** A body that delivers `chunks` as separate network reads. */
function body(chunks: string[]): ReadableStream<Uint8Array<ArrayBuffer>> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
}

async function collect(chunks: string[]) {
  const events = []
  for await (const e of parseSse(body(chunks))) events.push(e)
  return events
}

describe('parseSse', () => {
  it('parses several events delivered in one chunk', async () => {
    const events = await collect([
      'event: delta\ndata: {"text": "Hi "}\n\nevent: done\ndata: {}\n\n',
    ])
    expect(events).toEqual([
      { event: 'delta', data: { text: 'Hi ' } },
      { event: 'done', data: {} },
    ])
  })

  it('reassembles an event split across chunks', async () => {
    const events = await collect(['event: del', 'ta\ndata: {"te', 'xt": "café"}\n', '\n'])
    expect(events).toEqual([{ event: 'delta', data: { text: 'café' } }])
  })

  it('ignores an unterminated trailing block', async () => {
    const events = await collect(['event: done\ndata: {}\n\nevent: delta\ndata: {"text"'])
    expect(events).toEqual([{ event: 'done', data: {} }])
  })
})
