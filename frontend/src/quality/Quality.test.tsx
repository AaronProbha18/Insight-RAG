import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Quality as QualityData } from './api'
import { Quality } from './Quality'

const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
const measured = (result: 'PASS' | 'FAIL'): QualityData => ({
  state: 'measured',
  result,
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'fake:chat', embeddings: 'fake:embed', judge: 'fake:judge' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [],
  connectors: { pdf: { result, failed: [], metrics } },
})
const render = (initial?: QualityData) => renderToStaticMarkup(<Quality initial={initial} itemClassName="sidebar-item" />)

describe('Quality button', () => {
  it('is just "Quality" before any result, with the dialog closed', () => {
    const html = render()

    expect(html).toMatch(/<button[^>]*>.*Quality<\/button>/)
    expect(html).not.toContain('role="dialog"')
  })

  it.each([
    ['PASS', measured('PASS')],
    ['FAIL', measured('FAIL')],
    ['Not measured', { state: 'not_measured', eval_command: 'uv run plumbline-eval' } as QualityData],
  ])('names its state in text: %s', (state, initial) => {
    const html = render(initial)

    expect(html).toMatch(new RegExp(`<button[^>]*>.*Quality: ${state}</button>`))
    expect(html).toMatch(/<svg[^>]*aria-hidden="true"/)
  })
})

describe('Quality compact', () => {
  it('shows an icon and keeps the full label as screen-reader text', () => {
    const html = renderToStaticMarkup(<Quality initial={measured('PASS')} itemClassName="sidebar-item" compact />)

    expect(html).toContain('<svg')
    expect(html).toContain('<span class="sr-only">Quality: PASS</span>')
  })

  it('has a fallback icon before any result', () => {
    const html = renderToStaticMarkup(<Quality itemClassName="sidebar-item" compact />)

    expect(html).toContain('<svg')
    expect(html).toContain('<span class="sr-only">Quality</span>')
  })
})

describe('Quality trigger', () => {
  it('takes the item class, with an icon and the full label', () => {
    const html = render(measured('PASS'))

    expect(html).toMatch(/<button[^>]*class="[^"]*sidebar-item/)
    expect(html).toContain('<svg')
    expect(html).toContain('Quality: PASS')
    expect(html).not.toContain('sr-only')
  })
})
