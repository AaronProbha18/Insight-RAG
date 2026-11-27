import { beforeEach, describe, expect, it, vi } from 'vitest'
import { chatIdFor, currentPage, isPlainClick, navigate, pageFor, replace, subscribe } from './route'

// Node has no window, so stub the few parts of it the router touches.
function stubWindow() {
  const target = new EventTarget()
  const win = {
    location: { pathname: '/' },
    history: {
      pushState: vi.fn((_s: unknown, _t: string, path: string) => {
        win.location.pathname = path
      }),
      replaceState: vi.fn((_s: unknown, _t: string, path: string) => {
        win.location.pathname = path
      }),
    },
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
  }
  vi.stubGlobal('window', win)
  return win
}

describe('pageFor', () => {
  it('maps /app/documents to documents and everything else to chat', () => {
    expect(pageFor('/')).toBe('chat')
    expect(pageFor('/app/documents')).toBe('documents')
    expect(pageFor('/app/documents/')).toBe('documents')
    expect(pageFor('/app/unknown')).toBe('chat')
    expect(pageFor('/documents')).toBe('chat')
    expect(pageFor('/x')).toBe('chat')
  })

  it('maps /app/connectors to connectors, but not the API path or a sub-path', () => {
    expect(pageFor('/app/connectors')).toBe('connectors')
    expect(pageFor('/app/connectors/')).toBe('connectors')
    expect(pageFor('/connectors')).toBe('chat')
    expect(pageFor('/app/connectors/x')).toBe('chat')
    expect(pageFor('/app/constructor')).toBe('chat')
  })
})

describe('chat URLs', () => {
  it('names the saved chat in /app/chats/{id}, with or without a trailing slash', () => {
    expect(chatIdFor('/app/chats/conv_A')).toBe('conv_A')
    expect(chatIdFor('/app/chats/conv_A/')).toBe('conv_A')
    expect(chatIdFor('/app/chats/a%2Fb')).toBe('a/b')
  })

  it('names no chat for any other path', () => {
    for (const path of ['/', '/app/chats', '/app/chats/', '/app/chats/a/b', '/chats/conv_A', '/app/documents'])
      expect(chatIdFor(path)).toBeUndefined()
  })

  it('is the chat page, not a new page', () => {
    expect(pageFor('/app/chats/conv_A')).toBe('chat')
  })
})

describe('navigation', () => {
  let win: ReturnType<typeof stubWindow>
  beforeEach(() => {
    win = stubWindow()
  })

  it('navigate pushes history and notifies once', () => {
    const listener = vi.fn()
    subscribe(listener)

    navigate('/app/documents')

    expect(win.history.pushState).toHaveBeenCalledWith(null, '', '/app/documents')
    expect(currentPage()).toBe('documents')
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('replace changes the URL without a history entry and notifies once', () => {
    const listener = vi.fn()
    subscribe(listener)

    replace('/app/chats/conv_A')

    expect(win.history.replaceState).toHaveBeenCalledWith(null, '', '/app/chats/conv_A')
    expect(win.history.pushState).not.toHaveBeenCalled()
    expect(chatIdFor(win.location.pathname)).toBe('conv_A')
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it('navigates to connectors and back to chat on popstate', () => {
    navigate('/app/connectors')
    expect(currentPage()).toBe('connectors')

    win.location.pathname = '/'
    win.dispatchEvent(new Event('popstate'))

    expect(currentPage()).toBe('chat')
  })

  it('popstate (back) notifies and returns to chat', () => {
    const listener = vi.fn()
    navigate('/app/documents')
    subscribe(listener)

    win.location.pathname = '/'
    win.dispatchEvent(new Event('popstate'))

    expect(listener).toHaveBeenCalledTimes(1)
    expect(currentPage()).toBe('chat')
  })

  it('stops notifying after unsubscribe', () => {
    const listener = vi.fn()
    const unsubscribe = subscribe(listener)
    unsubscribe()

    navigate('/app/documents')
    win.dispatchEvent(new Event('popstate'))

    expect(listener).not.toHaveBeenCalled()
  })
})

describe('isPlainClick', () => {
  const click = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false }

  it('is true only for an unmodified left click', () => {
    expect(isPlainClick(click)).toBe(true)
    expect(isPlainClick({ ...click, button: 1 })).toBe(false)
    for (const key of ['metaKey', 'ctrlKey', 'shiftKey', 'altKey']) expect(isPlainClick({ ...click, [key]: true })).toBe(false)
  })
})
