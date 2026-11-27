import { useMemo, useState } from 'react'
import { AssistantRuntimeProvider, AuiConfig, Suggestions, useLocalRuntime } from '@assistant-ui/react'
import { Thread } from '@/components/assistant-ui/elements/thread.aui'
import type { Turn } from '@/conversations/api'
import { createChatAdapter } from './adapter'
import { turnsToMessages } from './restore'

// Welcome-screen suggestions: generic, so they fit any uploaded documents.
const generic = [
  { title: 'Summarise', label: 'the main points of my documents', prompt: 'Summarise the main points of my documents.' },
  { title: 'Find', label: 'key dates and deadlines', prompt: 'What key dates or deadlines do my documents mention?' },
  {
    title: 'Explain',
    label: 'a term used in my documents',
    prompt: 'Which technical terms do my documents use, and what do they mean?',
  },
]
const config = AuiConfig({ suggestions: Suggestions(generic) })
// After the sample documents are indexed, one question about them comes first.
const sampleConfig = AuiConfig({
  suggestions: Suggestions([
    { title: 'Try', label: 'the sample returns policy', prompt: 'What is the returns policy?' },
    ...generic,
  ]),
})

type ChatPaneProps = {
  samples?: boolean
  /** The saved chat this pane shows, with its turns. Read on mount: change `key` to switch chats. */
  conversationId?: string
  turns?: Turn[]
  /** Called once, with the id the server gave a new chat after its first answer. Captured on mount. */
  onConversation?: (id: string) => void
  /** Called after every answer or refusal, once it is recorded. Captured on mount. */
  onAnswer?: () => void
}

/** The chat. The runtime lives here, so remounting it (a new `key`) starts an empty thread. */
export function ChatPane({ samples = false, conversationId, turns, onConversation, onAnswer }: ChatPaneProps) {
  const [adapter] = useState(() => createChatAdapter({ id: conversationId, onId: onConversation, onAnswer }))
  const initialMessages = useMemo(() => (turns ? turnsToMessages(turns) : undefined), [turns])
  const runtime = useLocalRuntime(adapter, { initialMessages })
  return (
    <AssistantRuntimeProvider runtime={runtime} config={samples ? sampleConfig : config}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}
