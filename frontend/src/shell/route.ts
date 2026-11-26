import { useSyncExternalStore } from 'react'

export type Page = 'chat' | 'documents'

// Only /app/documents is a page; "/documents" is the API, so it stays chat.
export function pageFor(pathname: string): Page {
  return /^\/app\/documents\/?$/.test(pathname) ? 'documents' : 'chat'
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

export const useRoute = (): Page => useSyncExternalStore(subscribe, currentPage, () => 'chat')
