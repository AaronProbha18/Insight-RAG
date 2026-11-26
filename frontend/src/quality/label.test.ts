import { describe, expect, it } from 'vitest'
import type { Quality } from './api'
import { qualityLabel } from './label'

const measured = (result: 'PASS' | 'FAIL'): Quality => ({
  state: 'measured',
  result,
  run_at: '2026-11-25T09:00:00+00:00',
  models: { chat: 'c', embeddings: 'e', judge: 'j' },
  threshold: 0.3,
  config_matches: true,
  config_differences: [],
  connectors: {},
})

describe('qualityLabel', () => {
  it('says the overall state in words', () => {
    expect(qualityLabel(null)).toBe('Quality')
    expect(qualityLabel({ state: 'not_measured', eval_command: 'x' })).toBe('Quality: Not measured')
    expect(qualityLabel(measured('PASS'))).toBe('Quality: Passes')
    expect(qualityLabel(measured('FAIL'))).toBe('Quality: Below bar')
  })
})
