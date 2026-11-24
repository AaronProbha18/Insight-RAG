import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import {
  AssistantRuntimeProvider,
  ReadonlyThreadProvider,
  type ThreadMessage,
  useLocalRuntime,
} from '@assistant-ui/react'
import { chatAdapter } from '@/chat/adapter'
import { Thread } from './thread.aui'

const button = (name: string) =>
  new RegExp(`<button[^>]*>(?:(?!</button>).)*${name}(?:(?!</button>).)*</button>|<button[^>]*aria-label="${name}"`)

function Live() {
  return (
    <AssistantRuntimeProvider runtime={useLocalRuntime(chatAdapter)}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

// Static markup of a populated thread needs ReadonlyThreadProvider: `initialMessages`
// load after the first render. A read-only thread cannot cancel, so Stop while
// running is checked by hand (design D6).
// Synthetic conversation. Cast: only the fields the thread reads are filled in.
const conversation = (status: 'complete' | 'running', custom: object = {}) =>
  [
    {
      id: 'u1',
      role: 'user',
      createdAt: new Date(0),
      content: [{ type: 'text', text: 'What does the sample policy say?' }],
      attachments: [],
      metadata: { custom: {} },
    },
    {
      id: 'a1',
      role: 'assistant',
      createdAt: new Date(0),
      content: [{ type: 'text', text: 'Items can be returned within 14 days.' }],
      status: status === 'complete' ? { type: 'complete', reason: 'stop' } : { type: 'running' },
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom },
    },
  ] as unknown as ThreadMessage[]

function Readonly({ status, custom }: { status: 'complete' | 'running'; custom?: object }) {
  return (
    <AssistantRuntimeProvider runtime={useLocalRuntime(chatAdapter)}>
      <ReadonlyThreadProvider messages={conversation(status, custom)}>
        <Thread />
      </ReadonlyThreadProvider>
    </AssistantRuntimeProvider>
  )
}

describe('thread controls have accessible names', () => {
  it('shows scroll-to-bottom and Send, and no Stop, on an idle thread', () => {
    const html = renderToStaticMarkup(<Live />)

    expect(html).toMatch(button('Scroll to bottom'))
    expect(html).toMatch(button('Send message'))
    expect(html).not.toMatch(button('Stop generating'))
  })

  it('offers Copy and Regenerate on a completed answer', () => {
    const html = renderToStaticMarkup(<Readonly status="complete" />)

    expect(html).toContain('Items can be returned within 14 days.')
    expect(html).toMatch(button('Copy'))
    expect(html).toMatch(button('Regenerate'))
  })

  it('offers feedback on a recorded answer and on an abstention', () => {
    for (const abstained of [false, true]) {
      const html = renderToStaticMarkup(
        <Readonly status="complete" custom={{ citations: [], abstained, answerId: 'ans_1' }} />,
      )

      expect(html).toContain('Good answer')
      expect(html).toContain('Bad answer')
    }
  })

  it('offers no feedback while running or without an answer id', () => {
    const running = renderToStaticMarkup(<Readonly status="running" custom={{ answerId: 'ans_1' }} />)
    const unrecorded = renderToStaticMarkup(<Readonly status="complete" />)

    expect(running).not.toContain('Good answer')
    expect(unrecorded).not.toContain('Good answer')
  })
})
