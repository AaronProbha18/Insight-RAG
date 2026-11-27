import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ChatHeader } from './ChatHeader'

const render = (title: string, documentCount: number) => renderToStaticMarkup(<ChatHeader title={title} documentCount={documentCount} />)

describe('ChatHeader', () => {
  it('shows the title as the heading', () => {
    expect(render('Refund window', 3)).toMatch(/<h1[^>]*>Refund window<\/h1>/)
  })

  it('says how many documents the chat searches', () => {
    expect(render('t', 3)).toContain('Searching all 3 documents')
    expect(render('t', 1)).toContain('Searching all 1 document<')
    expect(render('t', 0)).toContain('Searching all 0 documents')
  })

  it('escapes the title', () => {
    const html = render('<script>x</script>', 1)

    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })

  it('is read-only: no control in it', () => {
    expect(render('t', 3)).not.toMatch(/<(button|input|select|textarea|a)\b/)
  })
})
