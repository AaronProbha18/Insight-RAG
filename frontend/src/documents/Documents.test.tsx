import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Documents } from './Documents'

describe('Documents', () => {
  it('renders a button named with the document count', () => {
    const html = renderToStaticMarkup(<Documents refreshKey={0} />)

    expect(html).toMatch(/<button[^>]*>Documents \(0\)<\/button>/)
    expect(html).not.toContain('role="dialog"')
  })
})
