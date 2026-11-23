import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Citations } from './Citations'

const cite = (id: string, title: string, locator: string, text: string) => ({
  id,
  document_id: `doc_${id}`,
  title,
  locator,
  text,
})

describe('Citations', () => {
  it('renders title, locator and cited text for each citation', () => {
    const html = renderToStaticMarkup(
      <Citations
        citations={[
          cite('c1', 'Returns Policy', 'p. 1', 'Most items can be returned within 14 days.'),
          cite('c2', 'Employee Handbook', 'Leave > Annual Leave', '25 days of annual leave.'),
        ]}
      />,
    )

    expect(html).toContain('aria-label="Sources"')
    expect(html.match(/<li[ >]/g)).toHaveLength(2)
    expect(html).toMatch(/<details><summary[^>]*>\[c1\] Returns Policy — p\. 1<\/summary>/)
    expect(html).toMatch(/<details><summary[^>]*>\[c2\] Employee Handbook — Leave &gt; Annual Leave<\/summary>/)
    expect(html).toMatch(/<blockquote[^>]*>Most items can be returned within 14 days\.<\/blockquote>/)
  })

  it('renders nothing for an empty list', () => {
    expect(renderToStaticMarkup(<Citations citations={[]} />)).toBe('')
  })

  it('escapes citation text', () => {
    const html = renderToStaticMarkup(<Citations citations={[cite('c1', 'T', 'p. 1', '<script>x</script>')]} />)

    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
  })
})
