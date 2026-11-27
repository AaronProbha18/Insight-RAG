import { describe, expect, it } from 'vitest'
import { adopt, newChat, syncTarget, type Target } from './chatTarget'

const empty: Target = { key: 0, adopted: false }

describe('syncTarget', () => {
  it('opens the saved chat named by the URL, on a reload or a direct link', () => {
    expect(syncTarget(empty, 'conv_A')).toEqual({ key: 1, id: 'conv_A', adopted: false })
  })

  it('keeps the same object while the URL still names the same chat', () => {
    const t: Target = { key: 3, id: 'conv_A', adopted: false }

    expect(syncTarget(t, 'conv_A')).toBe(t)
    expect(syncTarget(empty, undefined)).toBe(empty)
  })

  it('switches to another saved chat with a fresh pane', () => {
    const t: Target = { key: 3, id: 'conv_A', adopted: false }

    expect(syncTarget(t, 'conv_B')).toEqual({ key: 4, id: 'conv_B', adopted: false })
  })

  it('goes to an empty chat when the URL goes back to /', () => {
    expect(syncTarget({ key: 3, id: 'conv_A', adopted: false }, undefined)).toEqual({ key: 4, id: undefined, adopted: false })
  })
})

describe('adopt', () => {
  it('records the new chat id without remounting, and the URL change then matches', () => {
    const adopted = adopt({ key: 2, adopted: false }, 'conv_NEW')

    expect(adopted).toEqual({ key: 2, id: 'conv_NEW', adopted: true })
    expect(syncTarget(adopted, 'conv_NEW')).toBe(adopted)
  })
})

describe('newChat', () => {
  it('starts an empty thread with a fresh pane, from a saved or an adopted chat', () => {
    expect(newChat({ key: 2, id: 'conv_A', adopted: true })).toEqual({ key: 3, adopted: false })
    expect(syncTarget(newChat({ key: 2, id: 'conv_A', adopted: false }), undefined).id).toBeUndefined()
  })
})
