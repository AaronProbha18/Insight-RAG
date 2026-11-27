import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AssistantRuntimeProvider, ReadonlyThreadProvider, type ThreadMessage, useLocalRuntime } from '@assistant-ui/react'
import { Thread } from '@/components/assistant-ui/elements/thread.aui'
import type { Turn } from '@/conversations/api'
import { createChatAdapter } from './adapter'
import { turnsToMessages } from './restore'

const turns: Turn[] = [
  {
    answer_id: 'ans_1',
    question: 'Refund window?',
    answer: 'Refunds take 14 days [c1].',
    abstained: false,
    citations: [{ document_id: 'doc_1', title: 'Returns Policy', locator: 'p. 1', indexed: false }],
  },
  { answer_id: 'ans_2', question: 'Pet policy?', answer: 'Not covered.', abstained: true, citations: [] },
]

describe('turnsToMessages', () => {
  it('gives a user and an assistant message per turn, in order', () => {
    const messages = turnsToMessages(turns)

    expect(messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user', 'assistant'])
    expect(messages[0].content).toEqual([{ type: 'text', text: 'Refund window?' }])
    expect(messages[1].content).toEqual([{ type: 'text', text: 'Refunds take 14 days [c1].' }])
  })

  it('carries citations, abstained and answerId in metadata.custom', () => {
    const [, first, , second] = turnsToMessages(turns) as any[]

    expect(first.metadata.custom).toEqual({ citations: turns[0].citations, abstained: false, answerId: 'ans_1' })
    expect(second.metadata.custom).toEqual({ citations: [], abstained: true, answerId: 'ans_2' })
  })

  it('is accepted by useLocalRuntime as initialMessages (D7)', () => {
    function Probe() {
      useLocalRuntime(createChatAdapter(), { initialMessages: turnsToMessages(turns) })
      return null
    }

    expect(renderToStaticMarkup(<Probe />)).toBe('')
  })
})

// Static markup of a populated thread needs ReadonlyThreadProvider (initialMessages load after
// the first render), so this renders the same metadata the restored messages carry.
function Restored() {
  const messages = turnsToMessages(turns).map((m, i) => ({
    id: `m${i}`,
    createdAt: new Date(0),
    attachments: [],
    ...m,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], ...m.metadata },
  })) as unknown as ThreadMessage[]
  return (
    <AssistantRuntimeProvider runtime={useLocalRuntime(createChatAdapter())}>
      <ReadonlyThreadProvider messages={messages}>
        <Thread />
      </ReadonlyThreadProvider>
    </AssistantRuntimeProvider>
  )
}

describe('a restored thread', () => {
  it('shows the answers, their sources, the no-longer-indexed note and the feedback controls', () => {
    const html = renderToStaticMarkup(<Restored />)

    expect(html).toContain('Refund window?')
    expect(html).toContain('Refunds take 14 days')
    expect(html).toContain('aria-label="Sources"')
    expect(html).toContain('Returns Policy — p. 1')
    expect(html).toContain('Source no longer indexed')
    expect(html).toContain('Good answer')
  })
})
