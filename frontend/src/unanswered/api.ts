export type UnansweredItem = {
  question: string
  count: number
  last_asked_at: string
  reason: 'not_in_documents' | 'marked_wrong'
  feedback_reason: string | null
}

/** The day ranges the backend accepts; null is all time. */
export type Range = 7 | 30 | 90 | null

/** GET /unanswered. Throws on any failure; the caller shows a generic message. */
export async function listUnanswered(days: Range): Promise<UnansweredItem[]> {
  const response = await fetch(days === null ? '/unanswered' : `/unanswered?days=${days}`)
  if (!response.ok) throw new Error(`list unanswered failed (${response.status})`)
  return (await response.json()).items
}
