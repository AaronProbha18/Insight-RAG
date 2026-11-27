import { useCallback, useEffect, useState } from 'react'
import { type ConversationSummary, deleteConversation, listConversations, renameConversation } from './api'

/** The list fetch as data: the conversations, or null plus the message to show when it failed. */
export async function loadConversations(): Promise<{ conversations: ConversationSummary[] | null; message: string }> {
  try {
    return { conversations: await listConversations(), message: '' }
  } catch {
    return { conversations: null, message: "Couldn't load chats. Try again." }
  }
}

/** Renames one chat; returns '' on success (404 = already gone) or the message to show. */
export async function renameOutcome(id: string, title: string): Promise<string> {
  try {
    const result = await renameConversation(id, title)
    return result.kind === 'invalid' ? 'A title needs 1 to 200 characters.' : ''
  } catch {
    return "Couldn't rename the chat. Try again."
  }
}

/** Deletes one chat; returns '' on success (404 = already gone) or the message to show. */
export async function deleteOutcome(id: string): Promise<string> {
  try {
    await deleteConversation(id)
    return ''
  } catch {
    return "Couldn't delete the chat. Try again."
  }
}

/** The saved chats: fetched on mount and after refresh(), a rename or a delete. */
export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [message, setMessage] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [version, setVersion] = useState(0)
  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    let stale = false
    loadConversations().then((result) => {
      if (stale) return
      if (result.conversations) setConversations(result.conversations)
      setMessage(result.message)
      setLoaded(true)
    })
    return () => {
      stale = true
    }
  }, [version])

  async function rename(id: string, title: string) {
    const failure = await renameOutcome(id, title)
    if (failure) setMessage(failure)
    else refresh()
  }

  async function remove(id: string) {
    const failure = await deleteOutcome(id)
    if (failure) setMessage(failure)
    else refresh()
  }

  return { conversations, loaded, message, refresh, rename, remove }
}
