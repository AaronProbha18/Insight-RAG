import { CircleCheckIcon, CircleHelpIcon, CircleXIcon, FileTextIcon } from 'lucide-react'
import type { ConnectorQuality, Quality } from '../quality/api'
import type { Connector } from './api'

const GATE = 0.8 // mirrors GATE in evals/scoring.py; /quality doesn't send it
const METRICS = ['correctness', 'faithfulness', 'abstention'] as const
const DIFFERS = { chat: 'chat model', embeddings: 'embedding model', threshold: 'abstain threshold' }
const EVAL_COMMAND = 'uv run plumbline-eval' // the backend's own default, for a connector the summary lacks
const label = (m: string) => m[0].toUpperCase() + m.slice(1)
const pill = 'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-semibold'

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
  const subtitle = [
    connector.builtin ? 'Built-in' : null,
    connector.extensions.join(', '),
    `${documentCount} ${documentCount === 1 ? 'document' : 'documents'}`,
  ]
    .filter(Boolean)
    .join(' · ')
  return (
    <section aria-label={connector.name.toUpperCase()} className="space-y-4 rounded-xl border border-border p-5 text-sm">
      <div className="flex flex-wrap items-center gap-3">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-border">
          <FileTextIcon aria-hidden="true" className="size-[18px]" />
        </span>
        <div className="min-w-0 grow">
          <h2 className="text-base font-semibold">{connector.name.toUpperCase()}</h2>
          <p className="text-muted-foreground">{subtitle}</p>
        </div>
        {quality.state === 'measured' && entry ? (
          <Status result={entry.result} />
        ) : (
          <span className={`${pill} border-border text-muted-foreground`}>
            <CircleHelpIcon aria-hidden="true" className="size-3.5" />
            Not measured
          </span>
        )}
      </div>
      {quality.state === 'measured' && entry ? (
        <Measured quality={quality} entry={entry} />
      ) : (
        <p>
          Run <code>{quality.state === 'not_measured' ? quality.eval_command : EVAL_COMMAND}</code> to measure.
        </p>
      )}
    </section>
  )
}

function Status({ result }: { result: ConnectorQuality['result'] }) {
  const pass = result === 'PASS'
  const Icon = pass ? CircleCheckIcon : CircleXIcon
  return (
    <span className={`${pill} ${pass ? 'border-border text-foreground' : 'border-destructive text-destructive'}`}>
      <Icon aria-hidden="true" className="size-3.5" />
      {pass ? 'Passes' : 'Below bar'}
    </span>
  )
}

function Measured({
  quality,
  entry,
}: {
  quality: Extract<Quality, { state: 'measured' }>
  entry: ConnectorQuality
}) {
  const { models, config_matches, config_differences } = quality
  const { answerable_questions: answerable, unanswerable_questions: unanswerable } = entry
  const counts =
    typeof answerable === 'number' && typeof unanswerable === 'number'
      ? ` on ${answerable} answerable and ${unanswerable} unanswerable questions`
      : ''
  return (
    <>
      <ul className="space-y-3">
        {METRICS.map((m) => {
          const value = entry.metrics[m]
          const failed = entry.failed.includes(m)
          return (
            <li key={m} className="space-y-1.5">
              <p className="flex justify-between gap-2">
                <span className="text-muted-foreground">{label(m)}</span>
                <strong className={`font-semibold ${failed ? 'text-destructive' : ''}`}>
                  {value.toFixed(2)}
                  {failed && ` (below ${GATE.toFixed(2)})`}
                </strong>
              </p>
              <div aria-hidden="true" className="relative h-2 rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-foreground"
                  style={{ width: `${value * 100}%` }}
                />
                <div className="absolute inset-y-[-3px] w-0.5 bg-foreground" style={{ left: `${GATE * 100}%` }} />
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
      <p className="border-t border-border pt-3 text-xs text-muted-foreground">
        Last checked{' '}
        <time dateTime={quality.run_at}>{new Date(quality.run_at).toLocaleDateString(undefined, DATE)}</time>
        {counts} · Chat {models.chat} · Embeddings {models.embeddings} · Judged by {models.judge} · Abstain threshold{' '}
        {quality.threshold}
      </p>
    </>
  )
}

const DATE: Intl.DateTimeFormatOptions = { day: 'numeric', month: 'short', year: 'numeric' }
