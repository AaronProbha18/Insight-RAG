import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App accessible names', () => {
  it('names every control', () => {
    const html = renderToStaticMarkup(<App />)

    expect(html).toContain('Upload a PDF or Word document')
    expect(html).toContain('aria-label="Ask a question about your documents"')
    expect(html).toMatch(/<button[^>]*>Send<\/button>/)
  })
})
