import { AssistantRuntimeProvider, AuiConfig, Suggestions, useLocalRuntime } from '@assistant-ui/react'
import { Thread } from '@/components/assistant-ui/elements/thread.aui'
import { chatAdapter } from './adapter'

// Welcome-screen suggestions: generic, so they fit any uploaded documents.
const config = AuiConfig({
  suggestions: Suggestions([
    { title: 'Summarise', label: 'the main points of my documents', prompt: 'Summarise the main points of my documents.' },
    { title: 'Find', label: 'key dates and deadlines', prompt: 'What key dates or deadlines do my documents mention?' },
    {
      title: 'Explain',
      label: 'a term used in my documents',
      prompt: 'Which technical terms do my documents use, and what do they mean?',
    },
  ]),
})

/** The chat. The runtime lives here, so remounting it (a new `key`) starts an empty thread. */
export function ChatPane() {
  const runtime = useLocalRuntime(chatAdapter)
  return (
    <AssistantRuntimeProvider runtime={runtime} config={config}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}
