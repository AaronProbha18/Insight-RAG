import { Dialog } from '@base-ui/react/dialog'
import { ChartNoAxesColumnIcon, CircleCheckIcon, CircleHelpIcon, CircleXIcon } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { getQuality, type Quality as QualityData } from './api'
import { QualityPanel } from './QualityPanel'

const backdrop = 'fixed inset-0 bg-foreground/40'
const popup =
  'fixed left-1/2 top-1/2 w-[min(32rem,calc(100vw-2rem))] max-h-[80dvh] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg border border-border bg-background p-4 text-foreground shadow-lg'

function label(q: QualityData | null) {
  if (!q) return { text: 'Quality', Icon: null }
  if (q.state === 'not_measured') return { text: 'Quality: Not measured', Icon: CircleHelpIcon }
  return { text: `Quality: ${q.result}`, Icon: q.result === 'PASS' ? CircleCheckIcon : CircleXIcon }
}

/** "Quality: <state>" button; opens the latest eval result. `initial` seeds state for static tests.
 *  `compact` shows the icon only; the label stays as screen-reader text.
 *  `itemClassName` styles the trigger (the sidebar's item style). */
export function Quality({
  initial = null,
  compact = false,
  itemClassName,
}: {
  initial?: QualityData | null
  compact?: boolean
  itemClassName: string
}) {
  const [quality, setQuality] = useState<QualityData | null>(initial)
  const [open, setOpen] = useState(false)
  const [message, setMessage] = useState('')

  // Fetch on mount (for the button label) and each time the dialog opens.
  useEffect(() => {
    let stale = false
    getQuality().then(
      (q) => {
        if (stale) return
        setQuality(q)
        setMessage('')
      },
      () => !stale && setMessage("Couldn't load quality. Try again."),
    )
    return () => {
      stale = true
    }
  }, [open])

  const { text, Icon: Loaded } = label(quality)
  const Icon = Loaded ?? ChartNoAxesColumnIcon
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger className={itemClassName}>
        <Icon aria-hidden="true" className="size-4 shrink-0" />
        {compact ? <span className="sr-only">{text}</span> : text}
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Backdrop className={backdrop} />
        <Dialog.Popup className={popup}>
          <Dialog.Title className="mb-2 text-base font-semibold">Quality</Dialog.Title>
          {quality && <QualityPanel quality={quality} />}
          <p role="status" aria-live="polite" className="mt-2 min-h-5 text-sm text-destructive">
            {message}
          </p>
          <div className="mt-2 flex justify-end">
            <Dialog.Close render={<Button variant="outline" size="sm" />}>Close</Dialog.Close>
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
