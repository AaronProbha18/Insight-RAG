import { useEffect, useState } from 'react'
import { getQuality, type Quality } from './api'

/** Loads /quality; returns the data, or the message to show on any failure. */
export async function qualityOutcome(): Promise<{ quality: Quality } | { message: string }> {
  try {
    return { quality: await getQuality() }
  } catch {
    return { message: "Couldn't load quality. Try again." }
  }
}

/** The latest eval result: fetched on mount. `quality` is null until loaded. */
export function useQuality() {
  const [quality, setQuality] = useState<Quality | null>(null)
  const [message, setMessage] = useState('')

  useEffect(() => {
    let stale = false
    qualityOutcome().then((outcome) => {
      if (stale) return
      if ('quality' in outcome) {
        setQuality(outcome.quality)
        setMessage('')
      } else setMessage(outcome.message)
    })
    return () => {
      stale = true
    }
  }, [])

  return { quality, message }
}
