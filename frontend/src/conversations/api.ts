import type { Citation } from '../chat/adapter'

export type ConversationSummary = {
  conversation_id: string
  title: string
  updated_at: string
}

export type Turn = {
  answer_id: string
  question: string
  answer: string
  abstained: boolean
  citations: Citation[]
}

export type ConversationDetail = {
  conversation_id: string
  title: string
  turns: Turn[]
}

const path = (id: string) => `/conversations/${encodeURIComponent(id)}`

/** GET /conversations, newest first. Throws on any failure; the caller shows a generic message. */
export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await fetch('/conversations')
  if (!response.ok) throw new Error(`list conversations failed (${response.status})`)
  return response.json()
}

/** GET /conversations/{id}. `null` on 404; throws on any other failure. */
export async function getConversation(id: string): Promise<ConversationDetail | null> {
  const response = await fetch(path(id))
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`get conversation failed (${response.status})`)
  return response.json()
}

export type RenameResult = { kind: 'renamed'; title: string } | { kind: 'invalid' } | { kind: 'not-found' }

/** PATCH /conversations/{id}. 422 is `invalid`, 404 `not-found`; throws on anything else. */
export async function renameConversation(id: string, title: string): Promise<RenameResult> {
  const response = await fetch(path(id), {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (response.status === 422) return { kind: 'invalid' }
  if (response.status === 404) return { kind: 'not-found' }
  if (!response.ok) throw new Error(`rename conversation failed (${response.status})`)
  return { kind: 'renamed', title: (await response.json()).title }
}

/** DELETE /conversations/{id}. Throws on anything but 204 and 404. */
export async function deleteConversation(id: string): Promise<'deleted' | 'not-found'> {
  const response = await fetch(path(id), { method: 'DELETE' })
  if (response.status === 204) return 'deleted'
  if (response.status === 404) return 'not-found'
  throw new Error(`delete conversation failed (${response.status})`)
}
