import { useState } from 'react'
import { Button, buttonVariants } from '@/components/ui/button'
import { trySamples } from '../documents/useDocuments'
import { isPlainClick, navigate } from '../shell/route'

const STEPS = ['Add documents', 'Ask a question', 'Check the source']

type ViewProps = { error: string; busy: boolean; onUpload: () => void; onTry: () => void }

export function FirstRunView({ error, busy, onUpload, onTry }: ViewProps) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 p-6 text-center">
      <img src="/favicon.svg" alt="" className="size-10" />
      <h1 className="text-2xl font-semibold">Ask questions about your documents</h1>
      <p className="max-w-md text-muted-foreground">
        Every answer cites the passage it came from, so you can check it.
      </p>
      <ol className="flex list-decimal flex-col gap-1 text-left text-sm">
        {STEPS.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <div className="flex flex-wrap justify-center gap-2">
        <a
          href="/app/documents"
          onClick={(e) => {
            if (isPlainClick(e)) {
              e.preventDefault()
              onUpload()
            }
          }}
          className={buttonVariants()}
        >
          Upload documents
        </a>
        <Button variant="outline" disabled={busy} onClick={onTry}>
          Try the sample documents
        </Button>
      </div>
      <p role="status" className="min-h-5 text-sm">
        {error}
      </p>
    </div>
  )
}

/** The chat area when no documents are indexed. `onIndexed` runs after the samples are in. */
export function FirstRun({ onIndexed }: { onIndexed: () => void }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function onTry() {
    setBusy(true)
    const failure = await trySamples()
    setBusy(false)
    setError(failure)
    if (!failure) onIndexed()
  }

  return <FirstRunView error={error} busy={busy} onUpload={() => navigate('/app/documents')} onTry={onTry} />
}
