import { useState } from 'react'
import { REASONS } from '../feedback/reasons'
import { isPlainClick, navigate } from '../shell/route'
import type { Range, UnansweredItem } from './api'
import { useUnanswered } from './useUnanswered'

const RANGES: { days: Range; label: string }[] = [
  { days: 7, label: 'Last 7 days' },
  { days: 30, label: 'Last 30 days' },
  { days: 90, label: 'Last 90 days' },
  { days: null, label: 'All time' },
]

/** The reason as text: "Not in your documents", or "Marked wrong: wrong source". */
function reasonText(item: UnansweredItem): string {
  if (item.reason === 'not_in_documents') return 'Not in your documents'
  const label = REASONS.find((r) => r.value === item.feedback_reason)?.label
  return label ? `Marked wrong: ${label.toLowerCase()}` : 'Marked wrong'
}

type ViewProps = { items: UnansweredItem[]; message: string; days: Range; onDays: (days: Range) => void }

export function UnansweredView({ items, message, days, onDays }: ViewProps) {
  return (
    <div className="mx-auto w-full max-w-[44rem] px-4 py-6">
      <h1 className="mb-4 text-xl font-semibold">Unanswered</h1>
      <label className="flex items-center gap-2 text-sm font-medium">
        Date range
        <select
          value={String(days)}
          onChange={(e) => onDays(e.target.value === 'null' ? null : (Number(e.target.value) as Range))}
          className="rounded-md border border-border bg-background px-3 py-1.5 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        >
          {RANGES.map((r) => (
            <option key={String(r.days)} value={String(r.days)}>
              {r.label}
            </option>
          ))}
        </select>
      </label>
      <p role="status" aria-live="polite" className="mt-2 min-h-5 text-sm text-destructive">
        {message}
      </p>
      {items.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">
          Nothing here yet. Questions Plumbline can't answer, and answers marked wrong, will appear here.
        </p>
      ) : (
        <ul className="mt-2 divide-y divide-border">
          {items.map((item) => (
            <li key={item.question + item.last_asked_at} className="flex flex-col gap-1 py-3">
              <span className="font-medium">{item.question}</span>
              <span className="text-sm text-muted-foreground">
                {`Asked ${item.count} ${item.count === 1 ? 'time' : 'times'}`} · Last asked{' '}
                {new Date(item.last_asked_at).toLocaleDateString(undefined, { dateStyle: 'medium' })} · {reasonText(item)}
              </span>
              <a
                href="/app/documents"
                aria-label={`Add a document for: ${item.question}`}
                onClick={(e) => {
                  if (isPlainClick(e)) {
                    e.preventDefault()
                    navigate('/app/documents')
                  }
                }}
                className="w-fit text-sm font-medium underline underline-offset-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
              >
                Add a document
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Unanswered page: refused and wrong-marked questions for a range, default Last 30 days. */
export function UnansweredPage() {
  const [days, setDays] = useState<Range>(30)
  return <UnansweredView {...useUnanswered(days)} days={days} onDays={setDays} />
}
