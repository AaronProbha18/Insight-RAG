import { useState } from 'react'

const KEY = 'plumbline.sidebar.collapsed'

type Store = Pick<Storage, 'getItem' | 'setItem'>

// Storage can be missing or throw (private windows, blocked site data), even on access to
// `window.localStorage`, so callers pass a getter. The sidebar then starts expanded.
export function readCollapsed(storage: () => Store): boolean {
  try {
    return storage().getItem(KEY) === 'true'
  } catch {
    return false
  }
}

export function writeCollapsed(storage: () => Store, value: boolean): void {
  try {
    storage().setItem(KEY, String(value))
  } catch {
    // not persisted; ignore
  }
}

const local = () => window.localStorage

export function useCollapsed(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState(() => readCollapsed(local))
  const toggle = () => {
    writeCollapsed(local, !collapsed)
    setCollapsed(!collapsed)
  }
  return [collapsed, toggle]
}
