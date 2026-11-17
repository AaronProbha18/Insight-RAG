import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Upload } from './Upload'

describe('Upload', () => {
  it('has a labelled file input and a live status region', () => {
    const html = renderToStaticMarkup(<Upload />)

    expect(html).toMatch(/<label>Upload a PDF or Word document <input type="file" accept=".pdf,.docx"/)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-live="polite"')
  })
})
