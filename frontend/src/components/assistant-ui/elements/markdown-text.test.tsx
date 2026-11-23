import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { TextMessagePartProvider } from '@assistant-ui/react'
import { MarkdownText } from './markdown-text'

// `useLocalRuntime({ initialMessages })` loads messages after the first render,
// so static markup of the whole thread has none. Render one text part instead.
const render = (text: string) =>
  renderToStaticMarkup(
    <TextMessagePartProvider text={text} isRunning={false}>
      <MarkdownText />
    </TextMessagePartProvider>,
  )

describe('assistant answers render markdown', () => {
  it('formats lists, code blocks and tables', () => {
    const html = render(
      ['Steps:', '', '- one', '- two', '', '```py', 'print("hi")', '```', '', '| a | b |', '|---|---|', '| 1 | 2 |'].join(
        '\n',
      ),
    )

    expect(html).toMatch(/<ul[^>]*>\s*<li[^>]*>one<\/li>/)
    expect(html).toMatch(/<pre[^>]*>[\s\S]*<code[^>]*>[\s\S]*print/)
    expect(html).toMatch(/<table[^>]*>[\s\S]*<th[^>]*>a<\/th>[\s\S]*<td[^>]*>1<\/td>/)
    expect(html).not.toContain('|---|')
    expect(html).not.toContain('```')
  })

  it('does not turn raw HTML into elements', () => {
    const html = render('Before <script>alert(1)</script> and <b>x</b> after')

    expect(html).toContain('Before') // the part rendered at all
    expect(html).not.toMatch(/<script/i)
    expect(html).not.toMatch(/<b>/)
  })
})
