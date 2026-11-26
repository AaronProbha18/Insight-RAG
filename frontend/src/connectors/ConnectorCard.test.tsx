import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { Quality } from '../quality/api'
import type { Connector } from './api'
import { ConnectorCard } from './ConnectorCard'

const pdf: Connector = { name: 'pdf', extensions: ['.pdf'], contract_version: 1 }
const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
const measured = {
  state: 'measured' as const,
  result: 'PASS' as const,
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'fake:chat', embeddings: 'fake:embed', judge: 'fake:judge' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [] as ('chat' | 'embeddings' | 'threshold')[],
  connectors: { pdf: { result: 'PASS' as const, failed: [] as string[], metrics } },
}
const notMeasured: Quality = { state: 'not_measured', eval_command: 'uv run plumbline-eval' }
const render = (quality: Quality, documentCount = 3, connector = pdf) =>
  renderToStaticMarkup(<ConnectorCard connector={connector} documentCount={documentCount} quality={quality} />)

describe('ConnectorCard', () => {
  it('shows name, extensions and the document count', () => {
    const html = render(measured)

    expect(html).toContain('PDF')
    expect(html).toContain('.pdf')
    expect(html).toContain('3 documents')
    expect(render(measured, 1)).toContain('1 document<')
    expect(render(measured, 0)).toContain('0 documents')
  })

  it('shows Passes with three metrics, bars, the run date and model ids', () => {
    const html = render(measured)

    expect(html).toContain('Passes')
    expect(html).not.toContain('Below bar')
    for (const t of ['Correctness 0.90', 'Faithfulness 0.88', 'Abstention 0.95']) expect(html).toContain(t)
    expect(html).toContain('width:88%')
    expect(html).toContain('left:80%')
    expect(html).toContain(`<time dateTime="${measured.run_at}">${new Date(measured.run_at).toLocaleDateString()}</time>`)
    for (const id of ['fake:chat', 'fake:embed', 'fake:judge']) expect(html).toContain(id)
    expect(html).toContain('Abstain threshold 0.3')
    expect(html).not.toContain('not current')
  })

  it('shows Below bar and names only the failed metric', () => {
    const html = render({
      ...measured,
      result: 'FAIL',
      connectors: { pdf: { result: 'FAIL', failed: ['correctness'], metrics: { ...metrics, correctness: 0.7 } } },
    })

    expect(html).toContain('Below bar')
    expect(html).not.toContain('Passes')
    expect(html).toContain('Correctness 0.70 (below 0.80)')
    expect(html.match(/below 0\.80/g)).toHaveLength(1)
  })

  it('shows Not measured with the command and no numbers when there is no valid run', () => {
    const html = render(notMeasured)

    expect(html).toContain('Not measured')
    expect(html).toContain('<code>uv run plumbline-eval</code>')
    expect(html).not.toMatch(/\d\.\d\d/)
    expect(html).not.toContain('width:')
  })

  it('shows Not measured with the fallback command when the summary lacks this connector', () => {
    const html = render(measured, 2, { name: 'docx', extensions: ['.docx'], contract_version: 1 })

    expect(html).toContain('Not measured')
    expect(html).toContain('<code>uv run plumbline-eval</code>')
    expect(html).not.toContain('Passes')
    expect(html).not.toContain('Correctness')
  })

  it('says the result is not current when the configuration differs', () => {
    const html = render({ ...measured, config_matches: false, config_differences: ['chat', 'threshold'] })

    expect(html).toContain('role="status"')
    expect(html).toContain('This result is not current: different chat model, abstain threshold.')
  })

  it('never relies on colour alone: icon plus word, bars hidden, values in text', () => {
    for (const html of [render(measured), render(notMeasured)]) {
      expect(html).toMatch(/<svg[^>]*aria-hidden="true"/)
    }
    const html = render(measured)
    expect(html).toContain('aria-hidden="true" class="relative')
    expect(html).toContain('Correctness 0.90')
  })
})
