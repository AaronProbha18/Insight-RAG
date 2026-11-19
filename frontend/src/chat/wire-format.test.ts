import { describe, expect, it } from 'vitest'

// Guards docs/guides/frontend-development-guide.md §2: only the chat adapter
// speaks the /chat wire format, so a format change has exactly one place to go.
const sources = import.meta.glob(['/src/**/*.{ts,tsx}', '!/src/**/*.test.{ts,tsx}'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const owners = (pattern: RegExp) =>
  Object.entries(sources)
    .filter(([, code]) => pattern.test(code))
    .map(([path]) => path.replace(/^\/src\//, ''))
    .sort()

describe('chat wire format', () => {
  it('scans the source tree', () => {
    expect(Object.keys(sources).length).toBeGreaterThan(5)
  })

  it('only chat/adapter.ts calls /chat', () => {
    expect(owners(/['"`]\/chat['"`]/)).toEqual(['chat/adapter.ts'])
  })

  it('only chat/sse.ts parses SSE, and only the adapter uses the parser', () => {
    expect(owners(/TextDecoderStream|['"`]event: /)).toEqual(['chat/sse.ts'])
    const callers = owners(/\bparseSse\(/).filter((path) => path !== 'chat/sse.ts') // its definition
    expect(callers).toEqual(['chat/adapter.ts'])
  })
})
