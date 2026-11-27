import { useCallback, useEffect, useState } from 'react'
import { type DocumentSummary, deleteDocument, indexSamples, listDocuments } from './api'

/** Deletes one document; returns '' on success (404 = already gone) or the message to show. */
export async function deleteOutcome(id: string): Promise<string> {
  try {
    await deleteDocument(id)
    return ''
  } catch {
    return "Couldn't delete the document. Try again."
  }
}

const SAMPLES_FAILURE = "Couldn't index the sample documents. Try again."

/** Indexes the sample documents; returns '' when something was indexed, else the message to show. */
export async function trySamples(): Promise<string> {
  try {
    return (await indexSamples()).indexed.length > 0 ? '' : SAMPLES_FAILURE
  } catch {
    return SAMPLES_FAILURE
  }
}

/** The list fetch as data: the documents, or null plus the message to show when it failed. */
export async function loadDocuments(): Promise<{ documents: DocumentSummary[] | null; message: string }> {
  try {
    return { documents: await listDocuments(), message: '' }
  } catch {
    return { documents: null, message: "Couldn't load documents. Try again." }
  }
}

/** The indexed documents: fetched on mount and after refresh() or a delete. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [message, setMessage] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [version, setVersion] = useState(0)
  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    let stale = false
    loadDocuments().then((result) => {
      if (stale) return
      if (result.documents) setDocuments(result.documents)
      setMessage(result.message)
      setLoaded(true)
    })
    return () => {
      stale = true
    }
  }, [version])

  async function remove(d: DocumentSummary) {
    const failure = await deleteOutcome(d.document_id)
    if (failure) setMessage(failure)
    else refresh()
  }

  return { documents, loaded, message, refresh, remove }
}
