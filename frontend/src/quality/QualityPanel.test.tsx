import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Quality } from './api'
import { QualityPanel } from './QualityPanel'

const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
const base = {
  state: 'measured' as const,
  result: 'PASS' as const,
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'fake:chat', embeddings: 'fake:embed', judge: 'fake:judge' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [] as ('chat' | 'embeddings' | 'threshold')[],
  connectors: {
    pdf: { result: 'PASS' as const, failed: [], metrics },
    docx: { result: 'PASS' as const, failed: [], metrics },
  },
}
const render = (q: Quality) => renderToStaticMarkup(<QualityPanel quality={q} />)

describe('QualityPanel', () => {
  it('shows PASS, three metrics, the run date and model ids for each connector', () => {
    const html = render(base)

    expect(html).toContain('PDF PASS')
    expect(html).toContain('DOCX PASS')
    expect(html.match(/Correctness 0\.90 · Faithfulness 0\.88 · Abstention 0\.95/g)).toHaveLength(2)
    expect(html).toContain(`<time dateTime="${base.run_at}">${new Date(base.run_at).toLocaleDateString()}</time>`)
    for (const id of ['fake:chat', 'fake:embed', 'fake:judge']) expect(html).toContain(id)
    expect(html).toContain('Abstain threshold 0.3')
    expect(html).not.toContain('FAIL')
    expect(html).not.toContain('below')
    expect(html).not.toContain('different configuration')
  })

  it('names the failed metric as below 0.80', () => {
    const html = render({
      ...base,
      result: 'FAIL',
      connectors: {
        ...base.connectors,
        docx: { result: 'FAIL', failed: ['faithfulness'], metrics: { ...metrics, faithfulness: 0.7 } },
      },
    })

    expect(html).toContain('DOCX FAIL')
    expect(html).toContain('Faithfulness 0.70 (below 0.80)')
    expect(html).toContain('PDF PASS')
    expect(html.match(/below/g)).toHaveLength(1)
  })

  it('shows PASS and FAIL as words next to an aria-hidden icon, not colour alone', () => {
    const html = render({
      ...base,
      connectors: { ...base.connectors, docx: { result: 'FAIL', failed: ['faithfulness'], metrics } },
    })

    expect(html.match(/<svg[^>]*aria-hidden="true"/g)).toHaveLength(2)
    expect(html).toContain('PDF PASS')
    expect(html).toContain('DOCX FAIL')
  })

  it('warns about exactly the differing configuration', () => {
    const html = render({ ...base, config_matches: false, config_differences: ['chat', 'threshold'] })

    expect(html).toContain('role="status"')
    expect(html).toContain('different configuration: chat model, abstain threshold.')
    expect(html).not.toContain('embedding model')
  })

  it('says not measured, shows the command and no numbers', () => {
    const html = render({ state: 'not_measured', eval_command: 'uv run plumbline-eval' })

    expect(html).toContain('Not measured yet')
    expect(html).toContain('<code>uv run plumbline-eval</code>')
    expect(html).not.toMatch(/\d\.\d|PASS|FAIL/)
  })

  it('escapes connector names and model ids', () => {
    const evil = '<img src=x onerror=alert(1)>'
    const html = render({
      ...base,
      models: { ...base.models, chat: evil },
      connectors: { [evil]: base.connectors.pdf },
    })

    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
  })
})
