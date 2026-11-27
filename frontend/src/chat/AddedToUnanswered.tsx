import { isPlainClick, navigate } from '../shell/route'

/** After a recorded refusal: a link to the Unanswered page. Nothing for an answer or an unrecorded refusal. */
export function AddedToUnanswered({ abstained, answerId }: { abstained: boolean; answerId?: string }) {
  if (!abstained || !answerId) return null
  return (
    <a
      href="/app/unanswered"
      onClick={(e) => {
        if (isPlainClick(e)) {
          e.preventDefault()
          navigate('/app/unanswered')
        }
      }}
      className="mt-2 inline-block rounded-sm text-sm text-muted-foreground underline underline-offset-2 hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
    >
      Added to Unanswered
    </a>
  )
}
