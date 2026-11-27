/** Above the chat: which conversation this is, and what it searches. Read-only. */
export function ChatHeader({ title, documentCount }: { title: string; documentCount: number }) {
  return (
    <header className="shrink-0 border-b border-border px-4 py-2">
      <h1 className="truncate text-sm font-medium text-foreground">{title}</h1>
      <p className="text-xs text-muted-foreground">
        Searching all {documentCount} {documentCount === 1 ? 'document' : 'documents'}
      </p>
    </header>
  )
}
