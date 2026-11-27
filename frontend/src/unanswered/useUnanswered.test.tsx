import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { loadUnanswered, useUnanswered } from './useUnanswered'

afterEach(() => vi.unstubAllGlobals())

describe('loadUnanswered', () => {
  it('returns the items and no message on 200', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ items: [] }), { status: 200 })))
    expect(await loadUnanswered(30)).toEqual({ items: [], message: '' })
  })

  it('returns null and the message on a 500 or a network error', async () => {
    const failed = { items: null, message: "Couldn't load unanswered questions. Try again." }
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 500 })))
    expect(await loadUnanswered(30)).toEqual(failed)
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    expect(await loadUnanswered(30)).toEqual(failed)
  })
})

describe('useUnanswered', () => {
  it('starts empty and exposes refresh on the first render', () => {
    function Probe() {
      const { items, refresh } = useUnanswered(30)
      return <span>{`${items.length}:${typeof refresh}`}</span>
    }
    expect(renderToStaticMarkup(<Probe />)).toBe('<span>0:function</span>')
  })
})
