import { useCallback, useEffect, useState } from 'react'
import { type DocumentSummary, deleteDocument, listDocuments } from './api'

/** Deletes one document; returns '' on success (404 = already gone) or the message to show. */
export async function deleteOutcome(id: string): Promise<string> {
  try {
    await deleteDocument(id)
    return ''
  } catch {
    return "Couldn't delete the document. Try again."
  }
}

/** The indexed documents: fetched on mount and after refresh() or a delete. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [message, setMessage] = useState('')
  const [version, setVersion] = useState(0)
  const refresh = useCallback(() => setVersion((v) => v + 1), [])

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
  }, [version])

  async function remove(d: DocumentSummary) {
    const failure = await deleteOutcome(d.document_id)
    if (failure) setMessage(failure)
    else refresh()
  }

  return { documents, message, refresh, remove }
}
