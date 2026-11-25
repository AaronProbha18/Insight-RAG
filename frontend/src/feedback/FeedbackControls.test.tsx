import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { FeedbackView, type FeedbackState } from './FeedbackControls'

const noop = () => {}
const render = (state: FeedbackState) =>
  renderToStaticMarkup(<FeedbackView state={state} onUp={noop} onDown={noop} onCancel={noop} onSubmit={noop} />)

describe('FeedbackView', () => {
  it('idle: two named buttons, nothing pressed, no form', () => {
    const html = render({ kind: 'idle' })

    expect(html).toContain('Good answer')
    expect(html).toContain('Bad answer')
    expect(html).not.toContain('aria-pressed="true"')
    expect(html).not.toContain('<form')
  })

  it('choosing: reason group, labelled comment limited to 2000, send and cancel', () => {
    const html = render({ kind: 'choosing' })

    expect(html).toContain('<legend')
    for (const label of ['Wrong', 'Incomplete', 'Wrong source', 'Should have answered', 'Other']) {
      expect(html).toContain(label)
    }
    expect(html.match(/type="radio"/g)).toHaveLength(5)
    expect(html).toMatch(/Comment \(optional\)<textarea[^>]*maxLength="2000"/)
    expect(html).toContain('Send')
    expect(html).toContain('Cancel')
  })

  it('saved: recorded status and the chosen button pressed', () => {
    const html = render({ kind: 'saved', rating: 'up' })

    expect(html).toContain('role="status"')
    expect(html).toContain('Feedback recorded')
    expect(html.match(/aria-pressed="true"/g)).toHaveLength(1)
  })

  it('error: generic message only', () => {
    const html = render({ kind: 'error' })

    expect(html).toContain("Couldn&#x27;t save feedback. Try again.")
    expect(html).not.toContain('Feedback recorded')
  })
})
