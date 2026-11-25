import { AlertDialog } from '@base-ui/react/alert-dialog'
import { Dialog } from '@base-ui/react/dialog'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { type DocumentSummary, deleteDocument, listDocuments } from './api'
import { DocumentList } from './DocumentList'

const backdrop = 'fixed inset-0 bg-foreground/40'
const popup =
  'fixed left-1/2 top-1/2 w-[min(32rem,calc(100vw-2rem))] max-h-[80dvh] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-background p-4 text-foreground shadow-lg'

/** "Documents (n)" button; opens the list of indexed documents with delete + confirm. */
export function Documents({ refreshKey }: { refreshKey: number }) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [open, setOpen] = useState(false)
  const [pending, setPending] = useState<DocumentSummary | null>(null)
  const [message, setMessage] = useState('')

  const [version, setVersion] = useState(0) // bumped after a delete to refetch

  // Fetch on mount, after an upload (refreshKey), when the dialog opens, and after a delete.
  useEffect(() => {
    let stale = false
    listDocuments().then(
      (list) => {
        if (stale) return
        setDocuments(list)
        setMessage('')
      },
      () => !stale && setMessage("Couldn't load documents. Try again."),
    )
    return () => {
      stale = true
    }
  }, [refreshKey, open, version])

  async function confirmDelete() {
    if (!pending) return
    const target = pending
    setPending(null)
    try {
      await deleteDocument(target.document_id) // 'not-found' = already gone; refetch either way
      setVersion((v) => v + 1)
    } catch {
      setMessage("Couldn't delete the document. Try again.")
    }
  }

  return (
    <>
      <Dialog.Root open={open} onOpenChange={setOpen}>
        <Dialog.Trigger render={<Button variant="outline" size="sm" />}>Documents ({documents.length})</Dialog.Trigger>
        <Dialog.Portal>
          <Dialog.Backdrop className={backdrop} />
          <Dialog.Popup className={popup}>
            <Dialog.Title className="mb-2 text-base font-semibold">Documents</Dialog.Title>
            <DocumentList documents={documents} onDelete={setPending} />
            <p role="status" aria-live="polite" className="mt-2 min-h-5 text-sm text-destructive">
              {message}
            </p>
            <div className="mt-2 flex justify-end">
              <Dialog.Close render={<Button variant="outline" size="sm" />}>Close</Dialog.Close>
            </div>
          </Dialog.Popup>
        </Dialog.Portal>
      </Dialog.Root>

      <AlertDialog.Root open={pending !== null} onOpenChange={(next) => !next && setPending(null)}>
        <AlertDialog.Portal>
          <AlertDialog.Backdrop className={backdrop} />
          <AlertDialog.Popup className={popup}>
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
    </>
  )
}
