import { useState } from 'react'
import { type UploadOutcome, uploadDocument } from './api'

const FAILED: UploadOutcome['kind'][] = ['unsupported', 'unreadable', 'error']

export function Upload() {
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<UploadOutcome | null>(null)

  async function onChange(event: React.ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file) return
    setBusy(true)
    setOutcome(await uploadDocument(file))
    setBusy(false)
    input.value = '' // allow re-uploading the same file
  }

  const failed = !busy && outcome !== null && FAILED.includes(outcome.kind)

  return (
    <div className="border-b border-border">
      <div className="mx-auto w-full max-w-[44rem] px-4 py-3">
        <label className="flex flex-wrap items-center gap-2 text-sm font-medium">
          Upload a PDF or Word document{' '}
          <input
            type="file"
            accept=".pdf,.docx"
            onChange={onChange}
            disabled={busy}
            className="rounded-md text-sm text-muted-foreground file:me-3 file:rounded-md file:border file:border-border file:bg-background file:px-3 file:py-1 file:text-sm file:font-medium file:text-foreground hover:file:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:opacity-50"
          />
        </label>
        <p
          role="status"
          aria-live="polite"
          className={`mt-1 min-h-5 text-sm ${failed ? 'text-destructive' : 'text-muted-foreground'}`}
        >
          {busy ? 'Uploading…' : (outcome?.message ?? '')}
        </p>
      </div>
    </div>
  )
}
