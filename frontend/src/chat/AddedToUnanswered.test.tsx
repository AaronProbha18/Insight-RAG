import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AddedToUnanswered } from './AddedToUnanswered'

afterEach(() => vi.unstubAllGlobals())

describe('AddedToUnanswered', () => {
  it('links a recorded refusal to the Unanswered page', () => {
    const html = renderToStaticMarkup(<AddedToUnanswered abstained answerId="ans_1" />)
    expect(html).toContain('href="/app/unanswered"')
    expect(html).toContain('Added to Unanswered')
  })

  it('renders nothing for an answer or an unrecorded refusal', () => {
    expect(renderToStaticMarkup(<AddedToUnanswered abstained={false} answerId="ans_1" />)).toBe('')
    expect(renderToStaticMarkup(<AddedToUnanswered abstained />)).toBe('')
  })

  it('handles a plain click as navigation and leaves modified clicks alone', () => {
    const pushState = vi.fn()
    vi.stubGlobal('window', { history: { pushState }, location: { pathname: '/' }, addEventListener() {}, removeEventListener() {} })
    const link = AddedToUnanswered({ abstained: true, answerId: 'ans_1' }) as ReactElement<{ onClick: (e: object) => void }>
    const event = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, preventDefault: vi.fn() }
    link.props.onClick({ ...event, ctrlKey: true })
    expect(pushState).not.toHaveBeenCalled()
    link.props.onClick(event)
    expect(event.preventDefault).toHaveBeenCalled()
    expect(pushState).toHaveBeenCalledWith(null, '', '/app/unanswered')
  })
})
