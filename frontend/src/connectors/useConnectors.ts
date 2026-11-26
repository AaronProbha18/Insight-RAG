import { useEffect, useState } from 'react'
import { type Connector, listConnectors } from './api'

/** Loads /connectors; returns the list, or the message to show on any failure. */
export async function connectorsOutcome(): Promise<{ connectors: Connector[] } | { message: string }> {
  try {
    return { connectors: await listConnectors() }
  } catch {
    return { message: "Couldn't load connectors. Try again." }
  }
}

/** The loaded connectors, fetched on mount. `connectors` is null until loaded. */
export function useConnectors() {
  const [connectors, setConnectors] = useState<Connector[] | null>(null)
  const [message, setMessage] = useState('')

  useEffect(() => {
    let stale = false
    connectorsOutcome().then((outcome) => {
      if (stale) return
      if ('connectors' in outcome) setConnectors(outcome.connectors)
      else setMessage(outcome.message)
    })
    return () => {
      stale = true
    }
  }, [])

  return { connectors, message }
}
