export type Connector = {
  name: string
  extensions: string[]
  contract_version: number
}

/** GET /connectors, sorted by name. Throws on any failure; the caller shows a generic message. */
export async function listConnectors(): Promise<Connector[]> {
  const response = await fetch('/connectors')
  if (!response.ok) throw new Error(`list connectors failed (${response.status})`)
  return response.json()
}
