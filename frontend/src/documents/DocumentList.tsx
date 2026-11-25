import { Button } from '@/components/ui/button'
import type { DocumentSummary } from './api'

/** Indexed documents in the order given (newest first from the API). */
export function DocumentList({
  documents,
  onDelete,
}: {
  documents: DocumentSummary[]
  onDelete: (document: DocumentSummary) => void
}) {
  if (documents.length === 0) {
    return <p className="text-sm text-muted-foreground">No documents yet. Upload a PDF or DOCX to get started.</p>
  }
  return (
    <ul className="divide-y divide-border">
      {documents.map((d) => (
        <li key={d.document_id} className="flex items-center gap-3 py-2">
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{d.title}</p>
            <p className="text-sm text-muted-foreground">
              {d.connector.toUpperCase()} · <time dateTime={d.created_at}>{new Date(d.created_at).toLocaleDateString()}</time>
            </p>
          </div>
          <Button variant="ghost" size="sm" aria-label={`Delete ${d.title}`} onClick={() => onDelete(d)}>
            Delete
          </Button>
        </li>
      ))}
    </ul>
  )
}
