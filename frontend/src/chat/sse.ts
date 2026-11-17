export type SseEvent = { event: string; data: any }

/** Parses a text/event-stream body: named events, one JSON `data:` line each. */
export async function* parseSse(body: ReadableStream<BufferSource>): AsyncGenerator<SseEvent> {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += value
    let end
    // An event may arrive split across chunks; only parse complete blocks.
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const event = /^event: (.*)$/m.exec(block)?.[1] ?? 'message'
      const data = JSON.parse(/^data: (.*)$/m.exec(block)?.[1] ?? '{}')
      yield { event, data }
    }
  }
}
