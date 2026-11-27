import { useEffect, useState } from 'react'
import { type ConversationDetail, getConversation } from './api'

export type Loaded = { status: 'found'; conversation: ConversationDetail } | { status: 'not-found' } | { status: 'error' }
export type Saved = Loaded | { status: 'loading' }

/** One saved chat as data: found, not-found (404) or error (anything else). */
export async function loadConversation(id: string): Promise<Loaded> {
  try {
    const conversation = await getConversation(id)
    return conversation ? { status: 'found', conversation } : { status: 'not-found' }
  } catch {
    return { status: 'error' }
  }
}

/** The saved chat `id`, fetched when it changes; `loading` until then. Pass undefined to skip. */
export function useConversation(id: string | undefined): Saved {
  const [result, setResult] = useState<{ id: string; loaded: Loaded }>()

  useEffect(() => {
    if (!id) return
    let stale = false
    loadConversation(id).then((loaded) => {
      if (!stale) setResult({ id, loaded })
    })
    return () => {
      stale = true
    }
  }, [id])

  return result && result.id === id ? result.loaded : { status: 'loading' }
}
