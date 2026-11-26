import { afterEach, describe, expect, it, vi } from 'vitest'
import config from '../../vite.config'
import { listConnectors } from './api'

function respond(status: number, body?: object) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })),
  )
}

afterEach(() => vi.unstubAllGlobals())

const connectors = [
  { name: 'docx', extensions: ['.docx'], contract_version: 1, builtin: true },
  { name: 'pdf', extensions: ['.pdf'], contract_version: 1, builtin: true },
]

describe('listConnectors', () => {
  it('returns the connectors in the order given', async () => {
    respond(200, connectors)

    expect(await listConnectors()).toEqual(connectors)
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/connectors')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('returns an empty list', async () => {
    respond(200, [])

    expect(await listConnectors()).toEqual([])
  })

  it('throws on a server error', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })

    await expect(listConnectors()).rejects.toThrow('(500)')
  })

  it('throws on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))

    await expect(listConnectors()).rejects.toThrow()
  })
})

describe('dev proxy', () => {
  it('forwards /connectors to the backend', () => {
    expect(config.server?.proxy).toHaveProperty('/connectors')
  })
})
