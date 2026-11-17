import { useState } from 'react'
import { type UploadOutcome, uploadDocument } from './api'

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

  return (
    <div className="upload">
      <label>
        Upload a PDF or Word document{' '}
        <input type="file" accept=".pdf,.docx" onChange={onChange} disabled={busy} />
      </label>
      <p role="status" aria-live="polite" className={`upload-status ${outcome?.kind ?? ''}`}>
        {busy ? 'Uploading…' : (outcome?.message ?? '')}
      </p>
    </div>
  )
}
