import type { ReactElement, ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { UnansweredItem } from './api'
import { UnansweredView } from './UnansweredPage'

afterEach(() => vi.unstubAllGlobals())

const item = (over: Partial<UnansweredItem> = {}): UnansweredItem => ({
  question: 'What is the refund window?',
  count: 3,
  last_asked_at: '2026-11-20T10:00:00Z',
  reason: 'not_in_documents',
  feedback_reason: null,
  ...over,
})
const view = (items: UnansweredItem[], extra: { message?: string; days?: 7 | 30 | 90 | null } = {}) =>
  renderToStaticMarkup(<UnansweredView items={items} message={extra.message ?? ''} days={extra.days ?? 30} onDays={() => {}} />)

function find(node: ReactNode, match: (el: ReactElement<Record<string, any>>) => boolean): ReactElement<Record<string, any>> | null {
  if (!node || typeof node !== 'object') return null
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = find(child, match)
      if (found) return found
    }
    return null
  }
  const el = node as ReactElement<Record<string, any>>
  if (match(el)) return el
  return find(el.props?.children, match)
}

describe('UnansweredView', () => {
  it('shows the question, count, date and reason of a row', () => {
    const html = view([item(), item({ question: 'Pets?', count: 1 })])
    expect(html).toContain('What is the refund window?')
    expect(html).toContain('Asked 3 times')
    expect(html).toContain('Asked 1 time ')
    expect(html).toContain(new Date('2026-11-20T10:00:00Z').toLocaleDateString(undefined, { dateStyle: 'medium' }))
    expect(html).toContain('Not in your documents')
  })

  it('words a wrong-marked reason as text', () => {
    const wrong = item({ reason: 'marked_wrong', feedback_reason: 'wrong_source' })
    expect(view([wrong])).toContain('Marked wrong: wrong source')
    expect(view([{ ...wrong, feedback_reason: null }])).toContain('Marked wrong<')
  })

  it('links every row to the Documents page', () => {
    const html = view([item(), item({ question: 'Pets?' })])
    expect(html.match(/href="\/app\/documents"/g)).toHaveLength(2)
    expect(html).toContain('aria-label="Add a document for: Pets?"')
  })

  it('handles a plain click as navigation and leaves modified clicks alone', () => {
    const pushState = vi.fn()
    vi.stubGlobal('window', { history: { pushState }, location: { pathname: '/' }, addEventListener() {}, removeEventListener() {} })
    const tree = UnansweredView({ items: [item()], message: '', days: 30, onDays: () => {} })
    const link = find(tree, (el) => el.type === 'a')!
    const event = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, preventDefault: vi.fn() }
    link.props.onClick({ ...event, ctrlKey: true })
    expect(pushState).not.toHaveBeenCalled()
    link.props.onClick(event)
    expect(event.preventDefault).toHaveBeenCalled()
    expect(pushState).toHaveBeenCalledWith(null, '', '/app/documents')
  })

  it('offers the four ranges with the given one selected', () => {
    const html = view([], { days: 30 })
    for (const label of ['Last 7 days', 'Last 30 days', 'Last 90 days', 'All time']) expect(html).toContain(label)
    expect(html).toMatch(/<option value="30" selected="">Last 30 days/)
  })

  it('maps the select to a range', () => {
    const onDays = vi.fn()
    const tree = UnansweredView({ items: [], message: '', days: 30, onDays })
    const select = find(tree, (el) => el.type === 'select')!
    for (const [value, days] of [['7', 7], ['90', 90], ['null', null]] as const) {
      select.props.onChange({ target: { value } })
      expect(onDays).toHaveBeenLastCalledWith(days)
    }
  })

  it('shows the empty state only when there are no rows', () => {
    expect(view([])).toContain('Nothing here yet')
    expect(view([item()])).not.toContain('Nothing here yet')
  })

  it('shows a failure in a status region', () => {
    expect(view([], { message: "Couldn't load unanswered questions. Try again." })).toMatch(/role="status"[^>]*>Couldn(&#x27;|')t load/)
  })
})
