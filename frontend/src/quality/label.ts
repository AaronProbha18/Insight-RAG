import type { Quality } from './api'

/** The sidebar's Quality item text: the overall state in words. Plain "Quality" while unknown. */
export function qualityLabel(quality: Quality | null): string {
  if (!quality) return 'Quality'
  if (quality.state === 'not_measured') return 'Quality: Not measured'
  return `Quality: ${quality.result === 'PASS' ? 'Passes' : 'Below bar'}`
}
