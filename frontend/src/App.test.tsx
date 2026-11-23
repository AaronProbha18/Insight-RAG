import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App accessible names', () => {
  it('names every control', () => {
    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('Upload a PDF or Word document')
    expect(html).toMatch(/<textarea[^>]*aria-label="Ask a question about your documents"/)
    expect(html).toMatch(/<button[^>]*aria-label="Send message"/)
  })

  it('shows a welcome and suggested questions on an empty thread', () => {
    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('Ask about your documents')
    expect(html).toMatch(/<button[^>]*>(?:(?!<\/button>).)*Summarise(?:(?!<\/button>).)*<\/button>/)
  })
})
