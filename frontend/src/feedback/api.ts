export type Feedback = {
  answerId: string
  rating: 'up' | 'down'
  reason?: 'wrong' | 'incomplete' | 'wrong_source' | 'should_have_answered' | 'other'
  comment?: string
}

/** POST /feedback. Throws on anything but 204 and 404; the caller shows a generic message. */
export async function sendFeedback(f: Feedback): Promise<'saved' | 'not-found'> {
  const response = await fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer_id: f.answerId, rating: f.rating, reason: f.reason, comment: f.comment }),
  })
  if (response.status === 204) return 'saved'
  if (response.status === 404) return 'not-found'
  throw new Error(`send feedback failed (${response.status})`)
}
