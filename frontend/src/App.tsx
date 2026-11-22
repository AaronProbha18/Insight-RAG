import { AssistantRuntimeProvider, useLocalRuntime } from '@assistant-ui/react'
import { Thread } from '@/components/assistant-ui/elements/thread.aui'
import { chatAdapter } from './chat/adapter'
import { Upload } from './upload/Upload'

export default function App() {
  const runtime = useLocalRuntime(chatAdapter)
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-dvh flex-col">
        <Upload />
        <div className="min-h-0 flex-1">
          <Thread />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}
