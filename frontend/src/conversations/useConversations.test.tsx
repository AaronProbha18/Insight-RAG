import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { deleteOutcome, loadConversations, renameOutcome, useConversations } from './useConversations'

afterEach(() => vi.unstubAllGlobals())

function stubFetch(status: number, body?: object) {
  const fetchMock = vi.fn(async () => new Response(body === undefined ? null : JSON.stringify(body), { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function offline() {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
}

describe('loadConversations', () => {
  it('returns the conversations and no message on 200', async () => {
    const list = [{ conversation_id: 'conv_1', title: 'a', updated_at: '2026-01-01' }]
    stubFetch(200, list)

    expect(await loadConversations()).toEqual({ conversations: list, message: '' })
  })

  it('returns null and the message on a 500 or a network error', async () => {
    const failed = { conversations: null, message: "Couldn't load chats. Try again." }
    stubFetch(500)
    expect(await loadConversations()).toEqual(failed)
    offline()
    expect(await loadConversations()).toEqual(failed)
  })
})

describe('renameOutcome', () => {
  it('is empty after a rename and after an already-gone 404', async () => {
    stubFetch(200, { conversation_id: 'conv_1', title: 'x', updated_at: 'y' })
    expect(await renameOutcome('conv_1', 'x')).toBe('')
    stubFetch(404)
    expect(await renameOutcome('conv_1', 'x')).toBe('')
  })

  it('explains a 422', async () => {
    stubFetch(422)
    expect(await renameOutcome('conv_1', '')).toBe('A title needs 1 to 200 characters.')
  })

  it('returns the failure message on a 500 or a network error', async () => {
    stubFetch(500)
    expect(await renameOutcome('conv_1', 'x')).toBe("Couldn't rename the chat. Try again.")
    offline()
    expect(await renameOutcome('conv_1', 'x')).toBe("Couldn't rename the chat. Try again.")
  })
})

describe('deleteOutcome', () => {
  it('is empty after a delete and after an already-gone 404', async () => {
    stubFetch(204)
    expect(await deleteOutcome('conv_1')).toBe('')
    stubFetch(404)
    expect(await deleteOutcome('conv_1')).toBe('')
  })

  it('returns the failure message on a 500 or a network error', async () => {
    stubFetch(500)
    expect(await deleteOutcome('conv_1')).toBe("Couldn't delete the chat. Try again.")
    offline()
    expect(await deleteOutcome('conv_1')).toBe("Couldn't delete the chat. Try again.")
  })

  it('calls DELETE with the encoded id', async () => {
    const fetchMock = stubFetch(204)
    await deleteOutcome('conv_a/b')
    expect(fetchMock).toHaveBeenCalledWith('/conversations/conv_a%2Fb', { method: 'DELETE' })
  })
})

describe('useConversations', () => {
  it('starts with an empty list on the first render', () => {
    function Probe() {
      return <span>{useConversations().conversations.length}</span>
    }
    expect(renderToStaticMarkup(<Probe />)).toBe('<span>0</span>')
  })
})
