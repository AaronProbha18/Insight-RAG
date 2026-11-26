import type { ChatModelAdapter } from '@assistant-ui/react'
import { parseSse } from './sse'

export type Citation = { id: string; document_id: string; title: string; locator: string; text: string }

/** Streams POST /chat (wire format: design.md D8) into assistant-ui. */
export const chatAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortSignal,
      body: JSON.stringify({
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content.map((p) => (p.type === 'text' ? p.text : '')).join(''),
        })),
      }),
    })
    if (!response.ok || !response.body) throw new Error(`chat request failed (${response.status})`)

    let text = ''
    let citations: Citation[] = []
    let abstained = false
    let answerId: string | undefined
    for await (const { event, data } of parseSse(response.body)) {
      if (event === 'delta') text += data.text
      else if (event === 'citations') citations = data.citations
      else if (event === 'abstain') {
        text = data.message
        abstained = true
      } else if (event === 'error') throw new Error(data.message)
      else if (event === 'done') {
        answerId = data.answer_id
        break
      }
      yield { content: [{ type: 'text', text }], metadata: { custom: { citations, abstained } } }
    }
    // Last update carries the id the feedback controls need; error paths never reach it.
    yield { content: [{ type: 'text', text }], metadata: { custom: { citations, abstained, answerId } } }
  },
}
