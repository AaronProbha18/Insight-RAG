import { AssistantRuntimeProvider, AuiConfig, Suggestions, useLocalRuntime } from '@assistant-ui/react'
import { Thread } from '@/components/assistant-ui/elements/thread.aui'
import { chatAdapter } from './chat/adapter'
import { Upload } from './upload/Upload'

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

export default function App() {
  const runtime = useLocalRuntime(chatAdapter)
  return (
    <AssistantRuntimeProvider runtime={runtime} config={config}>
      <div className="flex h-dvh flex-col">
        <Upload />
        <div className="min-h-0 flex-1">
          <Thread />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}
