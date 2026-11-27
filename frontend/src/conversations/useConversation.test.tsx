import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { loadConversation, useConversation } from './useConversation'

afterEach(() => vi.unstubAllGlobals())

function stubFetch(status: number, body?: object) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status })))
}

describe('loadConversation', () => {
  it('returns the conversation with its turns on 200', async () => {
    const conversation = { conversation_id: 'conv_A', title: 'Refunds', turns: [] }
    stubFetch(200, conversation)

    expect(await loadConversation('conv_A')).toEqual({ status: 'found', conversation })
  })

  it('is not-found on 404', async () => {
    stubFetch(404, { detail: 'Conversation not found' })

    expect(await loadConversation('conv_X')).toEqual({ status: 'not-found' })
  })

  it('is an error on a 500 or a network failure', async () => {
    stubFetch(500)
    expect(await loadConversation('conv_A')).toEqual({ status: 'error' })
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    expect(await loadConversation('conv_A')).toEqual({ status: 'error' })
  })
})

describe('useConversation', () => {
  it('is loading on the first render', () => {
    function Probe() {
      return <span>{useConversation('conv_A').status}</span>
    }

    expect(renderToStaticMarkup(<Probe />)).toBe('<span>loading</span>')
  })
})
