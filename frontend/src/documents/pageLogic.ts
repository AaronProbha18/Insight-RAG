import { type UploadOutcome, uploadDocument } from '../upload/api'
import type { DocumentSummary } from './api'

export type TypeFilter = 'all' | 'pdf' | 'docx'

/** Title contains the query (any case), and the connector matches the type unless it is "all". */
export function filterDocuments(documents: DocumentSummary[], query: string, type: TypeFilter): DocumentSummary[] {
  const q = query.trim().toLowerCase()
  return documents.filter(
    (d) => d.title.toLowerCase().includes(q) && (type === 'all' || d.connector === type),
  )
}

/** Documents per connector name. */
export function countByConnector(documents: DocumentSummary[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const d of documents) counts[d.connector] = (counts[d.connector] ?? 0) + 1
  return counts
}

/** Uploads one file and asks for a list refresh only when it was newly indexed. */
export async function uploadAndRefresh(
  file: File,
  refresh: () => void,
  upload: typeof uploadDocument = uploadDocument,
): Promise<UploadOutcome> {
  const outcome = await upload(file)
  if (outcome.kind === 'indexed') refresh()
  return outcome
}
