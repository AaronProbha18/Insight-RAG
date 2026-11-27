import type { ReactElement, ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { FirstRunView } from './FirstRun'

afterEach(() => vi.unstubAllGlobals())

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

const FAILURE = 'Indexing failed. Try again.'

describe('FirstRunView', () => {
  const props = { error: '', busy: false, onUpload: vi.fn(), onTry: vi.fn() }
  const plain = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false }

  it('Upload is a link to the Documents page that navigates on a plain click only', () => {
    const onUpload = vi.fn()
    const preventDefault = vi.fn()
    const link = find(FirstRunView({ ...props, onUpload }), (el) => el.type === 'a')!
    expect(link.props.href).toBe('/app/documents')
    link.props.onClick({ ...plain, ctrlKey: true, preventDefault })
    expect(onUpload).not.toHaveBeenCalled()
    link.props.onClick({ ...plain, preventDefault })
    expect(onUpload).toHaveBeenCalledOnce()
    expect(preventDefault).toHaveBeenCalledOnce()
  })

  it('Try calls onTry and is disabled while busy', () => {
    const onTry = vi.fn()
    const isTry = (el: ReactElement<Record<string, any>>) => el.props.children === 'Try the sample documents'
    const button = find(FirstRunView({ ...props, onTry }), isTry)!
    button.props.onClick()
    expect(onTry).toHaveBeenCalledOnce()
    expect(find(FirstRunView({ ...props, busy: true }), isTry)!.props.disabled).toBe(true)
  })

  it('shows the headline, three steps and the error only when set', () => {
    const html = renderToStaticMarkup(<FirstRunView {...props} />)
    expect(html).toContain('Ask questions about your documents')
    expect(html.match(/<li /g)).toHaveLength(3)
    expect(html).not.toContain(FAILURE)
    expect(renderToStaticMarkup(<FirstRunView {...props} error={FAILURE} />)).toContain(FAILURE)
  })
})
