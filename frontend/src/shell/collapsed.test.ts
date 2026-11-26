import { describe, expect, it } from 'vitest'
import { readCollapsed, writeCollapsed } from './collapsed'

const memory = () => {
  const data = new Map<string, string>()
  return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) }
}
const broken = {
  getItem: () => {
    throw new Error('blocked')
  },
  setItem: () => {
    throw new Error('blocked')
  },
}

describe('collapsed storage', () => {
  it('round-trips and defaults to expanded', () => {
    const store = memory()
    expect(readCollapsed(() => store)).toBe(false)
    writeCollapsed(() => store, true)
    expect(readCollapsed(() => store)).toBe(true)
    writeCollapsed(() => store, false)
    expect(readCollapsed(() => store)).toBe(false)
  })

  it('swallows an error thrown while reaching storage itself', () => {
    const unreachable = () => {
      throw new Error('blocked')
    }
    expect(readCollapsed(unreachable)).toBe(false)
    expect(() => writeCollapsed(unreachable, true)).not.toThrow()
  })

  it('swallows storage errors', () => {
    expect(readCollapsed(() => broken)).toBe(false)
    expect(() => writeCollapsed(() => broken, true)).not.toThrow()
  })
})
