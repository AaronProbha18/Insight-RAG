export type DocumentSummary = {
  document_id: string
  title: string
  connector: string
  created_at: string
}

/** GET /documents, newest first. Throws on any failure; the caller shows a generic message. */
export async function listDocuments(): Promise<DocumentSummary[]> {
  const response = await fetch('/documents')
  if (!response.ok) throw new Error(`list documents failed (${response.status})`)
  return response.json()
}

/** DELETE /documents/{id}. Throws on anything but 204 and 404. */
export async function deleteDocument(id: string): Promise<'deleted' | 'not-found'> {
  const response = await fetch(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (response.status === 204) return 'deleted'
  if (response.status === 404) return 'not-found'
  throw new Error(`delete document failed (${response.status})`)
}

export type SampleResult = {
  indexed: { filename: string }[]
  failed: { filename: string; reason: string }[]
}

/** POST /documents/samples. Throws on any non-2xx; the caller shows a generic message. */
export async function indexSamples(): Promise<SampleResult> {
  const response = await fetch('/documents/samples', { method: 'POST' })
  if (!response.ok) throw new Error(`index samples failed (${response.status})`)
  return response.json()
}
