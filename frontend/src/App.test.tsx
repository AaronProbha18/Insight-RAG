import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ChatPane } from './chat/ChatPane'

describe('App accessible names', () => {
  it('names every chat control', () => {
    const html = renderToStaticMarkup(<ChatPane />)

    expect(html).toMatch(/<textarea[^>]*aria-label="Ask a question about your documents"/)
    expect(html).toMatch(/<button[^>]*aria-label="Send message"/)
  })

  it('shows a welcome and suggested questions on an empty thread', () => {
    const html = renderToStaticMarkup(<ChatPane />)

    expect(html).toContain('Ask about your documents')
    expect(html).toMatch(/<button[^>]*>(?:(?!<\/button>).)*Summarise(?:(?!<\/button>).)*<\/button>/)
  })

  it('has no upload bar above the chat', () => {
    const html = renderToStaticMarkup(<ChatPane />)

    expect(html).not.toContain('Upload a PDF or Word document')
    expect(html).not.toContain('type="file"')
  })

  it('offers the sample question only after the samples are indexed', () => {
    expect(renderToStaticMarkup(<ChatPane />)).not.toContain('What is the returns policy?')
    expect(renderToStaticMarkup(<ChatPane samples />)).toContain('the sample returns policy')
  })
})
