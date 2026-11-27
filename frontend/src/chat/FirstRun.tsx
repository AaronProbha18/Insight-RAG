import { UploadIcon } from 'lucide-react'
import { useState } from 'react'
import { Button, buttonVariants } from '@/components/ui/button'
import { trySamples } from '../documents/useDocuments'
import { isPlainClick, navigate } from '../shell/route'

const STEPS = [
  { title: 'Add documents', text: 'PDF or Word. They stay on your machine.' },
  { title: 'Ask a question', text: 'In your own words, like asking a colleague.' },
  { title: 'Check the source', text: 'Open the cited passage to see where the answer came from.' },
]

type ViewProps = { error: string; busy: boolean; onUpload: () => void; onTry: () => void }

export function FirstRunView({ error, busy, onUpload, onTry }: ViewProps) {
  return (
    <div className="flex h-full items-center justify-center overflow-y-auto p-10">
      <div className="flex w-full max-w-[40rem] flex-col items-center gap-5 text-center">
        <img src="/favicon.svg" alt="" className="size-14 shrink-0" />
        <h1 className="text-[28px] font-semibold tracking-[-0.02em]">Ask questions about your documents</h1>
        <p className="text-base leading-[1.6] text-muted-foreground">
          Every answer cites the page it came from, or says the answer isn&apos;t in your documents.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <a
            href="/app/documents"
            onClick={(e) => {
              if (isPlainClick(e)) {
                e.preventDefault()
                onUpload()
              }
            }}
            className={buttonVariants({ className: 'h-10 gap-2 px-4 font-semibold' })}
          >
            <UploadIcon aria-hidden="true" />
            Upload documents
          </a>
          <Button variant="outline" disabled={busy} onClick={onTry} className="h-10 gap-2 px-4 font-semibold">
            Try the sample documents
          </Button>
        </div>
        <p className="text-[13px] text-muted-foreground">
          The samples are four short documents from a fictional outdoor-gear shop.
        </p>
        <p role="status" className="text-sm text-destructive empty:hidden">
          {error}
        </p>
        <ol className="mt-3 flex w-full flex-wrap gap-3 text-left">
          {STEPS.map(({ title, text }, i) => (
            <li key={title} className="flex min-w-[180px] flex-1 flex-col gap-1.5 rounded-xl border border-border p-4">
              <span className="inline-flex size-6 items-center justify-center rounded-full bg-muted text-xs font-bold text-foreground">
                {i + 1}
              </span>
              <p className="text-sm font-semibold">{title}</p>
              <p className="text-[13px] leading-normal text-muted-foreground">{text}</p>
            </li>
          ))}
        </ol>
      </div>
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
