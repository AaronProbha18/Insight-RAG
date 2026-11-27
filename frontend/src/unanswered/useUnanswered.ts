import { useCallback, useEffect, useState } from 'react'
import { listUnanswered, type Range, type UnansweredItem } from './api'

/** The list fetch as data: the groups, or null plus the message to show when it failed. */
export async function loadUnanswered(days: Range): Promise<{ items: UnansweredItem[] | null; message: string }> {
  try {
    return { items: await listUnanswered(days), message: '' }
  } catch {
    return { items: null, message: "Couldn't load unanswered questions. Try again." }
  }
}

/** The unanswered groups for a range: fetched on mount, when `days` changes and after refresh(). */
export function useUnanswered(days: Range) {
  const [items, setItems] = useState<UnansweredItem[]>([])
  const [message, setMessage] = useState('')
  const [version, setVersion] = useState(0)
  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    let stale = false
    loadUnanswered(days).then((result) => {
      if (stale) return
      if (result.items) setItems(result.items)
      setMessage(result.message)
    })
    return () => {
      stale = true
    }
  }, [days, version])

  return { items, message, refresh }
}
