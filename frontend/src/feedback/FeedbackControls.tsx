import { useState } from 'react'
import { ThumbsDownIcon, ThumbsUpIcon } from 'lucide-react'
import { TooltipIconButton } from '@/components/assistant-ui/elements/tooltip-icon-button'
import { Button } from '@/components/ui/button'
import { sendFeedback, type Feedback } from './api'

type Reason = NonNullable<Feedback['reason']>
type Rating = Feedback['rating']

const REASONS: { value: Reason; label: string }[] = [
  { value: 'wrong', label: 'Wrong' },
  { value: 'incomplete', label: 'Incomplete' },
  { value: 'wrong_source', label: 'Wrong source' },
  { value: 'should_have_answered', label: 'Should have answered' },
  { value: 'other', label: 'Other' },
]

export type FeedbackState =
  | { kind: 'idle' }
  | { kind: 'choosing' }
  | { kind: 'saving' }
  | { kind: 'saved'; rating: Rating }
  | { kind: 'error' }

type ViewProps = {
  state: FeedbackState
  onUp: () => void
  onDown: () => void
  onCancel: () => void
  onSubmit: (reason: Reason | undefined, comment: string) => void
}

/** Stateless view: every state renders from data, so each is checkable as static markup. */
export function FeedbackView({ state, onUp, onDown, onCancel, onSubmit }: ViewProps) {
  const rated = state.kind === 'saved' ? state.rating : undefined
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1" role="group" aria-label="Rate this answer">
        <TooltipIconButton tooltip="Good answer" aria-pressed={rated === 'up'} onClick={onUp} disabled={state.kind === 'saving'}>
          <ThumbsUpIcon className={rated === 'up' ? 'fill-current' : undefined} />
        </TooltipIconButton>
        <TooltipIconButton
          tooltip="Bad answer"
          aria-pressed={rated === 'down' || state.kind === 'choosing'}
          onClick={onDown}
          disabled={state.kind === 'saving'}
        >
          <ThumbsDownIcon className={rated === 'down' ? 'fill-current' : undefined} />
        </TooltipIconButton>
        <span role="status" className="text-muted-foreground text-xs">
          {state.kind === 'saved' && 'Feedback recorded'}
          {state.kind === 'error' && "Couldn't save feedback. Try again."}
        </span>
      </div>
      {state.kind === 'choosing' && (
        <form
          className="border-border flex max-w-sm flex-col gap-2 rounded-md border p-3 text-sm"
          onSubmit={(e) => {
            e.preventDefault()
            const data = new FormData(e.currentTarget)
            onSubmit((data.get('reason') as Reason | null) ?? undefined, String(data.get('comment') ?? ''))
          }}
        >
          <fieldset className="flex flex-col gap-1">
            <legend className="text-foreground mb-1 font-medium">What went wrong? (optional)</legend>
            {REASONS.map((r) => (
              <label key={r.value} className="flex items-center gap-2">
                <input type="radio" name="reason" value={r.value} className="focus-visible:ring-ring/50 focus-visible:ring-2" />
                {r.label}
              </label>
            ))}
          </fieldset>
          <label className="flex flex-col gap-1">
            Comment (optional)
            <textarea
              name="comment"
              maxLength={2000}
              rows={3}
              className="border-input bg-background focus-visible:ring-ring/50 rounded-md border p-2 outline-none focus-visible:ring-2"
            />
          </label>
          <div className="flex gap-2">
            <Button type="submit" size="sm">
              Send
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </form>
      )}
    </div>
  )
}

/** Thumbs up/down for one recorded answer; posts to /feedback and shows the outcome. */
export function FeedbackControls({ answerId }: { answerId: string }) {
  const [state, setState] = useState<FeedbackState>({ kind: 'idle' })

  async function send(rating: Rating, reason?: Reason, comment?: string) {
    setState({ kind: 'saving' })
    try {
      const result = await sendFeedback({ answerId, rating, reason, comment: comment || undefined })
      setState(result === 'saved' ? { kind: 'saved', rating } : { kind: 'error' })
    } catch {
      setState({ kind: 'error' })
    }
  }

  return (
    <FeedbackView
      state={state}
      onUp={() => send('up')}
      onDown={() => setState({ kind: 'choosing' })}
      onCancel={() => setState({ kind: 'idle' })}
      onSubmit={(reason, comment) => send('down', reason, comment)}
    />
  )
}
