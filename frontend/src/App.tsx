import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
} from '@assistant-ui/react'
import { type Citation, chatAdapter } from './chat/adapter'
import { Citations } from './chat/Citations'
import { Upload } from './upload/Upload'

export default function App() {
  const runtime = useLocalRuntime(chatAdapter)
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="thread">
        <Upload />
        <ThreadPrimitive.Viewport className="viewport">
          <ThreadPrimitive.Messages>
            {({ message }) => (
              <MessagePrimitive.Root className={`message ${message.role}`}>
                <MessagePrimitive.Parts />
                {message.role === 'assistant' && (
                  <Citations citations={(message.metadata.custom?.citations as Citation[]) ?? []} />
                )}
              </MessagePrimitive.Root>
            )}
          </ThreadPrimitive.Messages>
        </ThreadPrimitive.Viewport>
        <ComposerPrimitive.Root className="composer">
          <ComposerPrimitive.Input
            className="input"
            aria-label="Ask a question about your documents"
            placeholder="Ask a question about your documents"
          />
          <ComposerPrimitive.Send className="send">Send</ComposerPrimitive.Send>
        </ComposerPrimitive.Root>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  )
}
