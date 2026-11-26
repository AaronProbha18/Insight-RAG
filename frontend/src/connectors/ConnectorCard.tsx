import { CircleCheckIcon, CircleHelpIcon, CircleXIcon } from 'lucide-react'
import type { ConnectorQuality, Quality } from '../quality/api'
import type { Connector } from './api'

const GATE = 0.8 // mirrors GATE in evals/scoring.py; /quality doesn't send it
const METRICS = ['correctness', 'faithfulness', 'abstention'] as const
const DIFFERS = { chat: 'chat model', embeddings: 'embedding model', threshold: 'abstain threshold' }
const EVAL_COMMAND = 'uv run plumbline-eval' // the backend's own default, for a connector the summary lacks
const label = (m: string) => m[0].toUpperCase() + m.slice(1)

/** One connector: what it reads, how many documents use it, and its latest quality result. Pure; the page fetches. */
export function ConnectorCard({
  connector,
  documentCount,
  quality,
}: {
  connector: Connector
  documentCount: number
  quality: Quality
}) {
  const entry = quality.state === 'measured' ? quality.connectors[connector.name] : undefined
  return (
    <section aria-label={connector.name.toUpperCase()} className="space-y-2 rounded-lg border border-border p-4 text-sm">
      <h2 className="text-base font-semibold">{connector.name.toUpperCase()}</h2>
      <p className="text-muted-foreground">
        {connector.extensions.join(', ')} · {documentCount} {documentCount === 1 ? 'document' : 'documents'}
      </p>
      {quality.state === 'measured' && entry ? (
        <Measured quality={quality} entry={entry} />
      ) : (
        <>
          <p className="flex items-center gap-2 font-medium">
            <CircleHelpIcon aria-hidden="true" className="size-4" />
            Not measured
          </p>
          <p>
            Run <code>{quality.state === 'not_measured' ? quality.eval_command : EVAL_COMMAND}</code> to measure.
          </p>
        </>
      )}
    </section>
  )
}

function Measured({
  quality,
  entry,
}: {
  quality: Extract<Quality, { state: 'measured' }>
  entry: ConnectorQuality
}) {
  const pass = entry.result === 'PASS'
  const Icon = pass ? CircleCheckIcon : CircleXIcon
  const { models, config_matches, config_differences } = quality
  return (
    <>
      <p className={`flex items-center gap-2 font-medium ${pass ? '' : 'text-destructive'}`}>
        <Icon aria-hidden="true" className="size-4" />
        {pass ? 'Passes' : 'Below bar'}
      </p>
      <ul className="space-y-1">
        {METRICS.map((m) => {
          const value = entry.metrics[m]
          return (
            <li key={m}>
              <span>
                {label(m)} {value.toFixed(2)}
                {entry.failed.includes(m) && ` (below ${GATE.toFixed(2)})`}
              </span>
              <div aria-hidden="true" className="relative mt-1 h-2 rounded bg-muted">
                <div className="h-full rounded bg-foreground" style={{ width: `${value * 100}%` }} />
                <div className="absolute inset-y-[-2px] w-0.5 bg-foreground" style={{ left: `${GATE * 100}%` }} />
              </div>
            </li>
          )
        })}
      </ul>
      {!config_matches && (
        <p role="status" className="font-medium text-destructive">
          This result is not current: different {config_differences.map((d) => DIFFERS[d]).join(', ')}.
        </p>
      )}
      <p className="text-muted-foreground">
        Run <time dateTime={quality.run_at}>{new Date(quality.run_at).toLocaleDateString()}</time> · Chat{' '}
        {models.chat} · Embeddings {models.embeddings} · Judge {models.judge} · Abstain threshold {quality.threshold}
      </p>
    </>
  )
}
