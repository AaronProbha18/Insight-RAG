import { AlertDialog } from '@base-ui/react/alert-dialog'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import type { UploadOutcome } from '../upload/api'
import type { DocumentSummary } from './api'
import { DocumentList } from './DocumentList'
import { type TypeFilter, filterDocuments, uploadAndRefresh } from './pageLogic'

const FAILED: UploadOutcome['kind'][] = ['unsupported', 'unreadable', 'error']
const field =
  'rounded-md border border-border bg-background px-3 py-1.5 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring'

type Props = {
  documents: DocumentSummary[]
  message: string
  refresh: () => void
  remove: (document: DocumentSummary) => Promise<void>
}

/** Documents page: upload, search by title, filter by type, list with delete + confirm. */
export function DocumentsPage({ documents, message, refresh, remove }: Props) {
  const [query, setQuery] = useState('')
  const [type, setType] = useState<TypeFilter>('all')
  const [busy, setBusy] = useState(false)
  const [outcome, setOutcome] = useState<UploadOutcome | null>(null)
  const [pending, setPending] = useState<DocumentSummary | null>(null)

  async function onUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file) return
    setBusy(true)
    setOutcome(await uploadAndRefresh(file, refresh))
    setBusy(false)
    input.value = '' // allow re-uploading the same file
  }

  function confirmDelete() {
    if (!pending) return
    const target = pending
    setPending(null)
    void remove(target)
  }

  const shown = filterDocuments(documents, query, type)
  const failed = !busy && outcome !== null && FAILED.includes(outcome.kind)

  return (
    <div className="mx-auto w-full max-w-[44rem] px-4 py-6">
      <h1 className="mb-4 text-xl font-semibold">Documents</h1>

      <label className="flex flex-wrap items-center gap-2 text-sm font-medium">
        Upload a PDF or Word document{' '}
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={onUpload}
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

      <div className="my-4 flex flex-wrap gap-4">
        <label className="flex items-center gap-2 text-sm font-medium">
          Search
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} className={field} />
        </label>
        <label className="flex items-center gap-2 text-sm font-medium">
          Type
          <select value={type} onChange={(e) => setType(e.target.value as TypeFilter)} className={field}>
            <option value="all">All</option>
            <option value="pdf">PDF</option>
            <option value="docx">Word</option>
          </select>
        </label>
      </div>

      {documents.length > 0 && shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No documents match.</p>
      ) : (
        <DocumentList documents={shown} onDelete={setPending} />
      )}
      <p role="status" aria-live="polite" className="mt-2 min-h-5 text-sm text-destructive">
        {message}
      </p>

      <AlertDialog.Root open={pending !== null} onOpenChange={(next) => !next && setPending(null)}>
        <AlertDialog.Portal>
          <AlertDialog.Backdrop className="fixed inset-0 bg-foreground/40" />
          <AlertDialog.Popup className="fixed left-1/2 top-1/2 w-[min(32rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-background p-4 text-foreground shadow-lg">
            <AlertDialog.Title className="text-base font-semibold">Delete "{pending?.title}"?</AlertDialog.Title>
            <AlertDialog.Description className="mt-1 text-sm text-muted-foreground">
              It will no longer be used in answers.
            </AlertDialog.Description>
            <div className="mt-4 flex justify-end gap-2">
              <AlertDialog.Close render={<Button variant="outline" size="sm" />}>Cancel</AlertDialog.Close>
              <Button variant="destructive" size="sm" onClick={confirmDelete}>
                Delete
              </Button>
            </div>
          </AlertDialog.Popup>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  )
}
