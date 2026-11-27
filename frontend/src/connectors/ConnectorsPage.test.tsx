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

    expect(html.match(/<section aria-label="(PDF|DOCX)"/g)).toHaveLength(2)
    expect(html).toContain('Built-in · .pdf · 3 documents')
    expect(html).toContain('Built-in · .docx · 0 documents')
    expect(html.indexOf('DOCX')).toBeLessThan(html.indexOf('PDF'))
  })

  it('has the title, intro and the 0.80 explainer with its link, in every state', () => {
    for (const html of [view(), view({ connectors: null }), view({ connectorsMessage: "Couldn't load connectors. Try again." })]) {
      expect(html).toContain('>Connectors and quality</h1>')
      expect(html).toContain('A connector reads one file type and turns it into passages Plumbline can cite.')
      expect(html).toContain('The bar is 0.80 on correctness, faithfulness and abstention.')
      expect(html).toMatch(/<a href="https:\/\/plumbline-docs.netlify.app\/architecture\/overview\/#evaluate-plumbline-eval"[^>]*rel="noreferrer"[^>]*>How it&#x27;s measured<\/a>/)
    }
  })

  it('ends with Build your own, linking to the connector guide, once loaded', () => {
    for (const html of [view(), view({ connectors: [] })]) {
      expect(html).toContain('<section aria-label="Build your own"')
      expect(html).toContain('href="https://plumbline-docs.netlify.app/guides/connector-development-guide/"')
      expect(html).toContain('Connector guide')
    }
    expect(view().indexOf('Build your own')).toBeGreaterThan(view().indexOf('PDF'))
  })

  it('shows no planned connectors, and no card for a quality entry that is not loaded', () => {
    const html = view({ connectors: [connectors[1]] })

    expect(html).not.toMatch(/Planned|Coming next/)
    expect(html.match(/<section aria-label="(PDF|DOCX)"/g)).toHaveLength(1)
    expect(html).not.toContain('aria-label="DOCX"')
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
      expect(html).not.toContain('Build your own')
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
      expect(html).not.toContain('Build your own')
      expect(html).not.toContain('Loading')
    }
  })

  it('says so when no connectors are loaded', () => {
    expect(view({ connectors: [] })).toContain('No connectors loaded.')
  })
})
