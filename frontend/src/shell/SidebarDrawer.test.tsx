import type { ReactElement, ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Sidebar } from './Sidebar'
import { SidebarDrawer } from './SidebarDrawer'

afterEach(() => vi.unstubAllGlobals())

// Sidebar has no hooks, so call it as a function and walk the element tree it returns.
function find(node: ReactNode, match: (el: ReactElement<Record<string, unknown>>) => boolean): ReactElement<Record<string, any>> | null {
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

const click = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, preventDefault: vi.fn() }

function setup(extra: Partial<Parameters<typeof Sidebar>[0]> = {}) {
  const pushState = vi.fn()
  vi.stubGlobal('window', { history: { pushState }, location: { pathname: '/' }, addEventListener() {}, removeEventListener() {} })
  const calls: string[] = []
  const tree = Sidebar({
    page: 'chat',
    documentCount: 1,
    collapsed: false,
    onNewChat: () => calls.push('newChat'),
    onNavigate: () => calls.push('navigate'),
    ...extra,
  })
  return { tree, pushState, calls }
}

describe('SidebarDrawer', () => {
  it('shows a named menu button only below md, with the drawer closed', () => {
    const html = renderToStaticMarkup(<SidebarDrawer page="chat" documentCount={0} onNewChat={() => {}} />)

    expect(html).toMatch(/<button[^>]*aria-label="Open menu"[^>]*class="[^"]*md:hidden|<button[^>]*class="[^"]*md:hidden[^"]*"[^>]*aria-label="Open menu"/)
    expect(html).not.toContain('role="dialog"')
  })
})

describe('Sidebar navigation callbacks', () => {
  it('a plain Documents click navigates and then closes', () => {
    const { tree, pushState, calls } = setup()
    const link = find(tree, (el) => el.props.href === '/app/documents')!

    link.props.onClick(click)

    expect(pushState).toHaveBeenCalledWith(null, '', '/app/documents')
    expect(calls).toEqual(['navigate'])
  })

  it('a ctrl-click closes without handling the navigation', () => {
    const { tree, pushState, calls } = setup()
    const link = find(tree, (el) => el.props.href === '/app/documents')!

    link.props.onClick({ ...click, ctrlKey: true })

    expect(pushState).not.toHaveBeenCalled()
    expect(calls).toEqual(['navigate'])
  })

  it('New chat runs first, then closes', () => {
    const { tree, calls } = setup()
    const button = find(tree, (el) => el.type === 'button')!

    button.props.onClick()

    expect(calls).toEqual(['newChat', 'navigate'])
  })

  it('has no collapse button without onToggleCollapse', () => {
    const without = renderToStaticMarkup(<Sidebar page="chat" documentCount={0} collapsed={false} onNewChat={() => {}} />)
    const withIt = renderToStaticMarkup(
      <Sidebar page="chat" documentCount={0} collapsed={false} onNewChat={() => {}} onToggleCollapse={() => {}} />,
    )

    expect(without).not.toContain('Collapse sidebar')
    expect(withIt).toContain('Collapse sidebar')
  })
})
