import type { DocumentSummary } from '../documents/api'
import { countByConnector } from '../documents/pageLogic'
import type { Quality } from '../quality/api'
import type { Connector } from './api'
import { ConnectorCard } from './ConnectorCard'
import { useConnectors } from './useConnectors'

type ViewProps = {
  connectors: Connector[] | null
  connectorsMessage: string
  quality: Quality | null
  qualityMessage: string
  documents: DocumentSummary[]
}

/** The page body; pure, so it can be rendered statically. */
export function ConnectorsView({ connectors, connectorsMessage, quality, qualityMessage, documents }: ViewProps) {
  const message = connectorsMessage || qualityMessage
  const counts = countByConnector(documents)
  let body = null
  if (!message && (!connectors || !quality)) body = 'Loading connectors…'
  else if (!message && connectors && connectors.length === 0) body = 'No connectors loaded.'

  return (
    <div className="mx-auto w-full max-w-[44rem] px-4 py-6">
      <h1 className="mb-4 text-xl font-semibold">Connectors</h1>
      <p role="status" aria-live="polite" className={`min-h-5 text-sm ${message ? 'text-destructive' : 'text-muted-foreground'}`}>
        {message || body}
      </p>
      {!message && connectors && quality && (
        <div className="space-y-3">
          {connectors.map((c) => (
            <ConnectorCard key={c.name} connector={c} documentCount={counts[c.name] ?? 0} quality={quality} />
          ))}
        </div>
      )}
    </div>
  )
}

/** Connectors page: one card per loaded connector, with its latest quality result. */
export function ConnectorsPage(props: Pick<ViewProps, 'quality' | 'documents'> & { qualityMessage: string }) {
  const { connectors, message } = useConnectors()
  return <ConnectorsView {...props} connectors={connectors} connectorsMessage={message} />
}
