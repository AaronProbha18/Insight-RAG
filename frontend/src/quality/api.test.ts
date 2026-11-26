import { afterEach, describe, expect, it, vi } from 'vitest'
import { getQuality, type Quality } from './api'

function respond(status: number, body?: object) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })),
  )
}

afterEach(() => vi.unstubAllGlobals())

const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
const measured: Quality = {
  state: 'measured',
  result: 'FAIL',
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'fake:chat', embeddings: 'fake:embed', judge: 'fake:judge' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [],
  connectors: {
    pdf: { result: 'PASS', failed: [], metrics },
    docx: { result: 'FAIL', failed: ['faithfulness'], metrics: { ...metrics, faithfulness: 0.7 } },
  },
}

describe('getQuality', () => {
  it('returns a measured result unchanged', async () => {
    respond(200, measured)

    expect(await getQuality()).toEqual(measured)
    const [url, init] = (fetch as any).mock.calls[0]
    expect(url).toBe('/quality')
    expect(init?.method ?? 'GET').toBe('GET')
  })

  it('passes the question counts through, and never invents them when absent', async () => {
    const withCounts: Quality = {
      ...measured,
      connectors: {
        pdf: { result: 'PASS', failed: [], metrics, answerable_questions: 30, unanswerable_questions: 10 },
      },
    }
    respond(200, withCounts)
    expect(await getQuality()).toEqual(withCounts)

    respond(200, measured)
    const got = await getQuality()
    expect(got.state === 'measured' && 'answerable_questions' in got.connectors.pdf).toBe(false)
  })

  it('returns not measured unchanged', async () => {
    const body = { state: 'not_measured', eval_command: 'uv run plumbline-eval' }
    respond(200, body)

    expect(await getQuality()).toEqual(body)
  })

  it('throws on a server error without leaking its body', async () => {
    respond(500, { detail: 'psycopg OperationalError at db:5432' })

    const error = await getQuality().catch((e: Error) => e)
    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).not.toContain('psycopg')
  })

  it('throws when the network fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))

    await expect(getQuality()).rejects.toThrow('offline')
  })
})
