import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { DocumentSummary } from '../documents/api'
import type { Quality } from '../quality/api'
import type { Connector } from './api'
import { countByConnector } from '../documents/pageLogic'
import { ConnectorsView } from './ConnectorsPage'

const connectors: Connector[] = [
  { name: 'docx', extensions: ['.docx'], contract_version: 1, builtin: true },
  { name: 'pdf', extensions: ['.pdf'], contract_version: 1, builtin: true },
]
const doc = (connector: string, i: number): DocumentSummary => ({
  document_id: `doc_${i}`,
  title: `T${i}`,
  connector,
  created_at: '2026-11-01T10:00:00Z',
})
const documents = [doc('pdf', 1), doc('pdf', 2), doc('pdf', 3)]
const metrics = { correctness: 0.9, faithfulness: 0.88, abstention: 0.95 }
const measured: Quality = {
  state: 'measured',
  result: 'FAIL',
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'fake:chat', embeddings: 'fake:embed', judge: 'fake:judge' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [],
  connectors: {
    pdf: { result: 'PASS', failed: [], metrics },
    docx: { result: 'FAIL', failed: ['correctness'], metrics: { ...metrics, correctness: 0.7 } },
  },
}
const view = (over: Partial<Parameters<typeof ConnectorsView>[0]> = {}) =>
  renderToStaticMarkup(
    <ConnectorsView
      connectors={connectors}
      connectorsMessage=""
      quality={measured}
      qualityMessage=""
      documents={documents}
      {...over}
    />,
  )

describe('countByConnector', () => {
  it('counts documents per connector name', () => {
    expect(countByConnector(documents)).toEqual({ pdf: 3 })
    expect(countByConnector([])).toEqual({})
  })
})

describe('ConnectorsView', () => {
  it('shows a card per connector with its extensions and document count', () => {
    const html = view()

    expect(html).toContain('<h1 class="mb-4 text-xl font-semibold">Connectors</h1>')
    expect(html.match(/<section/g)).toHaveLength(2)
    expect(html).toContain('.pdf · 3 documents')
    expect(html).toContain('.docx · 0 documents')
    expect(html.indexOf('DOCX')).toBeLessThan(html.indexOf('PDF'))
  })

  it('joins quality by connector name', () => {
    const html = view()

    expect(html).toContain('Passes')
    expect(html).toContain('Below bar')
    expect(html).toContain('0.70 (below 0.80)')
  })

  it('shows Not measured on every card when there is no valid run', () => {
    const html = view({ quality: { state: 'not_measured', eval_command: 'uv run plumbline-eval' } })

    expect(html.match(/Not measured/g)).toHaveLength(2)
    expect(html).not.toContain('Passes')
  })

  it('shows a loading line and no cards until both reads arrive', () => {
    for (const html of [view({ connectors: null }), view({ quality: null })]) {
      expect(html).toContain('Loading connectors…')
      expect(html).not.toContain('<section')
    }
  })

  it('shows only the message when a read fails', () => {
    for (const over of [
      { connectorsMessage: "Couldn't load connectors. Try again." },
      { quality: null, qualityMessage: "Couldn't load quality. Try again." },
    ]) {
      const html = view(over)
      expect(html).toContain("Couldn&#x27;t load")
      expect(html).not.toContain('<section')
      expect(html).not.toContain('Loading')
    }
  })

  it('says so when no connectors are loaded', () => {
    expect(view({ connectors: [] })).toContain('No connectors loaded.')
  })
})
