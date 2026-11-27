import { useSyncExternalStore } from 'react'

export type Page = 'chat' | 'documents' | 'connectors' | 'unanswered'

// Pages live under /app/; "/documents" and "/connectors" are the API, so they stay chat.
const pages: Record<string, Page> = { '/app/documents': 'documents', '/app/connectors': 'connectors', '/app/unanswered': 'unanswered' }

export function pageFor(pathname: string): Page {
  return pages[pathname.replace(/\/$/, '')] ?? 'chat'
}

/** The saved chat a path names (`/app/chats/{id}`), or undefined for any other path. */
export function chatIdFor(pathname: string): string | undefined {
  const id = /^\/app\/chats\/([^/]+)\/?$/.exec(pathname)?.[1]
  return id && decodeURIComponent(id)
}

const listeners = new Set<() => void>()

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  window.addEventListener('popstate', listener)
  return () => {
    listeners.delete(listener)
    window.removeEventListener('popstate', listener)
  }
}

export const currentPage = (): Page => pageFor(window.location.pathname)

export function navigate(path: string): void {
  window.history.pushState(null, '', path)
  listeners.forEach((l) => l())
}

/** Changes the URL without adding a history entry. */
export function replace(path: string): void {
  window.history.replaceState(null, '', path)
  listeners.forEach((l) => l())
}

/** A plain left-click, which the router handles; modified clicks are left to the browser. */
export const isPlainClick = (e: { button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean }) =>
  e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey

export const useRoute = (): Page => useSyncExternalStore(subscribe, currentPage, () => 'chat')

export const useChatId = (): string | undefined =>
  useSyncExternalStore(subscribe, () => chatIdFor(window.location.pathname), () => undefined)
