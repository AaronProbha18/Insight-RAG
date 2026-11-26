import type { DocumentSummary } from '../documents/api'
import { countByConnector } from '../documents/pageLogic'
import type { Quality } from '../quality/api'
import type { Connector } from './api'
import { ConnectorCard } from './ConnectorCard'
import { useConnectors } from './useConnectors'

const REPO = 'https://github.com/AaronProbha18/Plumbline/blob/main'
const MEASURED_URL = `${REPO}/evals/README.md`
const GUIDE_URL = `${REPO}/docs/guides/connector-development-guide.md`
const link =
  'rounded-sm font-medium text-foreground underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring'

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

  const loaded = !message && connectors && quality
  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6">
      <header className="space-y-1">
        <h1 className="text-xl font-semibold">Connectors and quality</h1>
        <p className="text-sm text-muted-foreground">
          A connector reads one file type and turns it into passages Plumbline can cite. Each one is checked against
          its own question set.
        </p>
      </header>
      <p className="flex flex-wrap items-center gap-2.5 text-sm">
        <span aria-hidden="true" className="h-3.5 w-0.5 bg-foreground" />
        The bar is 0.80 on correctness, faithfulness and abstention. A connector below it still works, but its answers
        are less reliable.
        <a href={MEASURED_URL} target="_blank" rel="noreferrer" className={link}>
          How it's measured
        </a>
      </p>
      <p role="status" aria-live="polite" className={`min-h-5 text-sm ${message ? 'text-destructive' : 'text-muted-foreground'}`}>
        {message || body}
      </p>
      {loaded && (
        <>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(21rem,1fr))] gap-5">
            {connectors.map((c) => (
              <ConnectorCard key={c.name} connector={c} documentCount={counts[c.name] ?? 0} quality={quality} />
            ))}
          </div>
          <section aria-label="Build your own" className="space-y-1.5 rounded-xl border border-border p-4 text-sm">
            <h2 className="text-base font-semibold">Build your own</h2>
            <p className="text-muted-foreground">
              Connectors are Python plug-ins. To be listed as supported, one must pass the conformance kit and reach
              the 0.80 bar.
            </p>
            <a href={GUIDE_URL} target="_blank" rel="noreferrer" className={link}>
              Connector guide
            </a>
          </section>
        </>
      )}
    </div>
  )
}

/** Connectors page: one card per loaded connector, with its latest quality result. */
export function ConnectorsPage(props: Pick<ViewProps, 'quality' | 'documents'> & { qualityMessage: string }) {
  const { connectors, message } = useConnectors()
  return <ConnectorsView {...props} connectors={connectors} connectorsMessage={message} />
}
