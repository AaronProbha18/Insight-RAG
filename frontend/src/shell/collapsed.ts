import { useState } from 'react'

const KEY = 'plumbline.sidebar.collapsed'

type Store = Pick<Storage, 'getItem' | 'setItem'>

// Storage can be missing or throw (private windows, blocked site data); the sidebar then starts expanded.
export function readCollapsed(storage: Store): boolean {
  try {
    return storage.getItem(KEY) === 'true'
  } catch {
    return false
  }
}

export function writeCollapsed(storage: Store, value: boolean): void {
  try {
    storage.setItem(KEY, String(value))
  } catch {
    // not persisted; ignore
  }
}

export function useCollapsed(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return readCollapsed(window.localStorage)
    } catch {
      return false
    }
  })
  const toggle = () => {
    try {
      writeCollapsed(window.localStorage, !collapsed)
    } catch {
      // window.localStorage itself can throw; the toggle still works for this session
    }
    setCollapsed(!collapsed)
  }
  return [collapsed, toggle]
}
