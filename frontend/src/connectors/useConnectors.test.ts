import { afterEach, describe, expect, it, vi } from 'vitest'
import { connectorsOutcome } from './useConnectors'

afterEach(() => vi.unstubAllGlobals())

const message = "Couldn't load connectors. Try again."

describe('connectorsOutcome', () => {
  it('returns the list', async () => {
    const list = [{ name: 'pdf', extensions: ['.pdf'], contract_version: 1 }]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(list), { status: 200 })))

    expect(await connectorsOutcome()).toEqual({ connectors: list })
  })

  it('returns only the generic message on a server error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'psycopg at db:5432' }), { status: 500 })),
    )

    expect(await connectorsOutcome()).toEqual({ message })
  })

  it('returns the generic message on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))

    expect(await connectorsOutcome()).toEqual({ message })
  })
})
