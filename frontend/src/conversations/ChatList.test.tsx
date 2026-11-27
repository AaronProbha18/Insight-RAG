import type { ReactElement, ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { ChatList, ChatRowView, type ChatsProps } from './ChatList'
import { saveAction } from './saveAction'

const chat = { conversation_id: 'conv_A', title: 'Refund window', updated_at: '2026-11-02T10:00:00Z' }
const other = { conversation_id: 'conv_B', title: 'Pet policy', updated_at: '2026-11-01T10:00:00Z' }

const chats = (extra: Partial<ChatsProps> = {}): ChatsProps => ({
  items: [chat, other],
  message: '',
  onRename: () => {},
  onDelete: () => {},
  ...extra,
})

const handlers = { onStartRename() {}, onStartDelete() {}, onCancel() {}, onSave() {}, onConfirmDelete() {} }
const row = (mode: 'idle' | 'rename' | 'delete', extra = {}) =>
  renderToStaticMarkup(<ChatRowView chat={chat} current={false} mode={mode} {...handlers} {...extra} />)

// ChatRowView has no hooks, so call it as a function and walk the element it returns.
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
  return match(el) ? el : find(el.props?.children, match)
}

describe('ChatList', () => {
  it('renders nothing when there are no chats', () => {
    expect(renderToStaticMarkup(<ChatList chats={chats({ items: [] })} />)).toBe('')
  })

  it('lists the chats under a Chats heading, in the order given, as links', () => {
    const html = renderToStaticMarkup(<ChatList chats={chats()} />)

    expect(html).toContain('>Chats</h2>')
    expect(html.indexOf('Refund window')).toBeLessThan(html.indexOf('Pet policy'))
    expect(html).toContain('href="/app/chats/conv_A"')
    expect(html).toContain('href="/app/chats/conv_B"')
  })

  it('announces only the open chat as the current page', () => {
    const html = renderToStaticMarkup(<ChatList chats={chats({ currentId: 'conv_B' })} />)
    const links = html.match(/<a\b[^>]*>/g) ?? []

    expect(links).toHaveLength(2)
    expect(links.filter((l) => l.includes('aria-current="page"'))).toEqual([expect.stringContaining('/app/chats/conv_B')])
    expect(renderToStaticMarkup(<ChatList chats={chats()} />)).not.toContain('aria-current')
  })

  it('names every control: the link by its title, Rename and Delete by the chat they act on', () => {
    const html = renderToStaticMarkup(<ChatList chats={chats()} />)

    for (const title of ['Refund window', 'Pet policy']) {
      expect(html).toContain(`aria-label="Rename ${title}"`)
      expect(html).toContain(`aria-label="Delete ${title}"`)
    }
    expect(html).not.toMatch(/tabindex="-1"|aria-disabled|disabled=""/)
  })

  it('shows a failure message from the last rename or delete', () => {
    const html = renderToStaticMarkup(<ChatList chats={chats({ message: "Couldn't delete the chat. Try again." })} />)

    expect(html).toMatch(/<p role="status"[^>]*>Couldn&#x27;t delete the chat\. Try again\.<\/p>/)
    expect(renderToStaticMarkup(<ChatList chats={chats()} />)).not.toContain('role="status"')
  })
})

describe('ChatRowView modes', () => {
  it('rename: an input holding the title, with Save and Cancel', () => {
    const html = row('rename')

    expect(html).toMatch(/<input[^>]*aria-label="Chat title"[^>]*value="Refund window"|<input[^>]*value="Refund window"[^>]*aria-label="Chat title"/)
    expect(html).toContain('maxLength="200"')
    expect(html).toContain('>Save<')
    expect(html).toContain('>Cancel<')
    expect(html).not.toContain('href=')
  })

  it('delete: asks first, with Delete and Cancel', () => {
    const html = row('delete')

    expect(html).toContain('Delete this chat?')
    expect(html).toContain('>Delete<')
    expect(html).toContain('>Cancel<')
    expect(html).not.toContain('href=')
  })

  it('Delete in the confirmation calls onConfirmDelete; Cancel calls onCancel', () => {
    const onConfirmDelete = vi.fn()
    const onCancel = vi.fn()
    const tree = ChatRowView({ chat, current: false, mode: 'delete', ...handlers, onConfirmDelete, onCancel })

    find(tree, (el) => el.props.children === 'Delete' && !!el.props.onClick)!.props.onClick()
    find(tree, (el) => el.props.children === 'Cancel')!.props.onClick()

    expect(onConfirmDelete).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('the row buttons start a rename and a delete', () => {
    const onStartRename = vi.fn()
    const onStartDelete = vi.fn()
    const tree = ChatRowView({ chat, current: false, mode: 'idle', ...handlers, onStartRename, onStartDelete })

    find(tree, (el) => el.props['aria-label'] === 'Rename Refund window')!.props.onClick()
    find(tree, (el) => el.props['aria-label'] === 'Delete Refund window')!.props.onClick()

    expect(onStartRename).toHaveBeenCalledTimes(1)
    expect(onStartDelete).toHaveBeenCalledTimes(1)
  })

  it('submitting the rename form saves the entered title', () => {
    const onSave = vi.fn()
    const tree = ChatRowView({ chat, current: false, mode: 'rename', ...handlers, onSave })
    const form = find(tree, (el) => typeof el.props.onSubmit === 'function')!
    const preventDefault = vi.fn()
    const currentTarget = { elements: { namedItem: (name: string) => (name === 'title' ? { value: '  Returns  ' } : null) } }

    form.props.onSubmit({ preventDefault, currentTarget })

    expect(preventDefault).toHaveBeenCalled()
    expect(onSave).toHaveBeenCalledWith('  Returns  ') // ChatList trims it through saveAction
  })
})

describe('saveAction', () => {
  it('cancels an empty, blank or unchanged title', () => {
    for (const entered of ['', '   ', 'Refund window', ' Refund window '])
      expect(saveAction('Refund window', entered)).toEqual({ kind: 'cancel' })
  })

  it('saves a changed title, trimmed', () => {
    expect(saveAction('Refund window', '  Returns  ')).toEqual({ kind: 'save', title: 'Returns' })
  })
})
