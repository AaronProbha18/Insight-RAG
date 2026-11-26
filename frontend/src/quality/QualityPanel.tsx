import { CircleCheckIcon, CircleXIcon } from 'lucide-react'
import type { Quality } from './api'

const GATE = 0.8 // mirrors GATE in evals/scoring.py; /quality doesn't send it
const METRICS = ['correctness', 'faithfulness', 'abstention'] as const
const DIFFERS = { chat: 'chat model', embeddings: 'embedding model', threshold: 'abstain threshold' }
const label = (m: string) => m[0].toUpperCase() + m.slice(1)

/** The latest eval result per connector; pure, the dialog does the fetching. */
export function QualityPanel({ quality }: { quality: Quality }) {
  if (quality.state === 'not_measured') {
    return (
      <p className="text-sm">
        Not measured yet. Run <code>{quality.eval_command}</code> to measure.
      </p>
    )
  }
  const { models, connectors, config_matches, config_differences } = quality
  return (
    <div className="space-y-3 text-sm">
      <p className="text-muted-foreground">
        Run <time dateTime={quality.run_at}>{new Date(quality.run_at).toLocaleDateString()}</time> · Chat{' '}
        {models.chat} · Embeddings {models.embeddings} · Judge {models.judge} · Abstain threshold {quality.threshold}
      </p>
      {!config_matches && (
        <p role="status" className="font-medium text-destructive">
          This result is for a different configuration: {config_differences.map((d) => DIFFERS[d]).join(', ')}.
        </p>
      )}
      <ul className="divide-y divide-border">
        {Object.entries(connectors).map(([name, c]) => {
          const Icon = c.result === 'PASS' ? CircleCheckIcon : CircleXIcon
          return (
            <li key={name} className="py-2">
              <p className={`flex items-center gap-2 font-medium ${c.result === 'FAIL' ? 'text-destructive' : ''}`}>
                <Icon aria-hidden="true" className="size-4" />
                {name.toUpperCase()} {c.result}
              </p>
              <p className="text-muted-foreground">
                {METRICS.map((m) => `${label(m)} ${c.metrics[m].toFixed(2)}${c.failed.includes(m) ? ` (below ${GATE.toFixed(2)})` : ''}`).join(' · ')}
              </p>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
