import { UploadIcon } from 'lucide-react'
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
    <div className="flex min-h-full flex-col">
      <header className="flex flex-wrap items-center gap-4 border-b border-border px-4 pt-7 pb-5 md:px-10">
        <div className="min-w-60 grow">
          <h1 className="text-[22px] font-semibold tracking-tight">Unanswered questions</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Questions Plumbline couldn't answer, and answers people marked wrong. Each one points to a document you may be missing.
          </p>
        </div>
        <label>
          <span className="sr-only">Date range</span>
          <select
            value={String(days)}
            onChange={(e) => onDays(e.target.value === 'null' ? null : (Number(e.target.value) as Range))}
            className="min-h-10 rounded-lg border border-border bg-background px-4 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          >
            {RANGES.map((r) => (
              <option key={String(r.days)} value={String(r.days)}>
                {r.label}
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="flex flex-col gap-4 px-4 pt-6 pb-10 md:px-10">
        <p role="status" aria-live="polite" className="min-h-5 text-sm text-destructive">
          {message}
        </p>
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing here yet. Questions Plumbline can't answer, and answers marked wrong, will appear here.
          </p>
        ) : (
          <ul className="divide-y divide-border rounded-xl border border-border">
            {items.map((item) => (
              <li key={item.question + item.last_asked_at} className="flex flex-wrap items-center gap-x-5 gap-y-3 px-5 py-4">
                <div className="min-w-0 flex-[1_1_20rem]">
                  <p className="text-[15px] font-semibold">{item.question}</p>
                  <p className="mt-1.5 flex flex-wrap items-center gap-2 text-[13px] text-muted-foreground">
                    {`Asked ${item.count} ${item.count === 1 ? 'time' : 'times'}`} · last{' '}
                    {new Date(item.last_asked_at).toLocaleDateString(undefined, { dateStyle: 'medium' })}
                    <span className="inline-flex items-center rounded-full bg-muted px-2 py-[3px] text-xs font-semibold text-foreground">
                      {reasonText(item)}
                    </span>
                  </p>
                </div>
                <a
                  href="/app/documents"
                  aria-label={`Add a document for: ${item.question}`}
                  onClick={(e) => {
                    if (isPlainClick(e)) {
                      e.preventDefault()
                      navigate('/app/documents')
                    }
                  }}
                  className="inline-flex min-h-9 items-center gap-2 rounded-lg border border-border bg-background px-3 text-sm font-medium hover:bg-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
                >
                  <UploadIcon aria-hidden="true" className="size-3.5 shrink-0" />
                  Add a document
                </a>
              </li>
            ))}
          </ul>
        )}
        <p className="text-[13px] text-muted-foreground">
          Questions are grouped when they match after ignoring case and spacing. Nothing here leaves your machine.
        </p>
      </div>
    </div>
  )
}

/** Unanswered page: refused and wrong-marked questions for a range, default Last 30 days. */
// ponytail: no loaded flag, so the empty state flashes before the first response; add `loaded` like useDocuments if it matters
export function UnansweredPage() {
  const [days, setDays] = useState<Range>(30)
  return <UnansweredView {...useUnanswered(days)} days={days} onDays={setDays} />
}
