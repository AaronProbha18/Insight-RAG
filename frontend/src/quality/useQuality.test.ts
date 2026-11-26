import { afterEach, describe, expect, it, vi } from 'vitest'
import { qualityOutcome } from './useQuality'

afterEach(() => vi.unstubAllGlobals())

function respond(status: number, body: object) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const message = "Couldn't load quality. Try again."

describe('qualityOutcome', () => {
  it('returns a measured result', async () => {
    const body = { state: 'measured', result: 'PASS', connectors: {} }
    const fetchMock = respond(200, body)

    expect(await qualityOutcome()).toEqual({ quality: body })
    expect(fetchMock).toHaveBeenCalledWith('/quality')
  })

  it('returns not_measured as data, not an error', async () => {
    const body = { state: 'not_measured', eval_command: 'uv run plumbline-eval' }
    respond(200, body)

    expect(await qualityOutcome()).toEqual({ quality: body })
  })

  it('returns only the generic message on a server error', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })

    expect(await qualityOutcome()).toEqual({ message })
  })

  it('returns the generic message on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')))

    expect(await qualityOutcome()).toEqual({ message })
  })
})
