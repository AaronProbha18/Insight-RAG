/// <reference types="node" /> // this test reads index.css from disk
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// Guards the styling rules in docs/guides/frontend-development-guide.md §6.
const css = readFileSync(fileURLToPath(new URL('./index.css', import.meta.url)), 'utf-8')
const COLOUR = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/

/** Text-on-background pairs the stylesheet actually uses. */
const PAIRS: [text: string, background: string][] = [
  ['--fg', '--bg'], // page text
  ['--fg', '--muted'], // assistant messages
  ['--on-accent', '--accent'], // user messages, Send button
  ['--error', '--bg'], // upload errors
]

function tokenBlocks(source: string): { light: string; dark: string; rest: string } {
  const light = /^:root\s*{([^}]*)}/m.exec(source)
  const dark = /@media \(prefers-color-scheme: dark\)\s*{\s*:root\s*{([^}]*)}\s*}/.exec(source)
  if (!light || !dark) throw new Error('index.css must define :root tokens and a dark-mode :root block')
  return { light: light[1], dark: dark[1], rest: source.replace(light[0], '').replace(dark[0], '') }
}

function tokens(block: string): Record<string, string> {
  return Object.fromEntries([...block.matchAll(/(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b/g)].map((m) => [m[1], m[2]]))
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

describe('design tokens', () => {
  const { light, dark, rest } = tokenBlocks(css)

  it('has no colour literals outside the two token blocks', () => {
    const stray = rest.split('\n').filter((line) => COLOUR.test(line.replace(/\/\*.*?\*\//g, '')))
    expect(stray).toEqual([])
  })

  it.each([
    ['light', light],
    ['dark', dark],
  ])('meets WCAG AA (4.5:1) for every text/background pair in %s mode', (_, block) => {
    const t = tokens(block)
    for (const [fg, bg] of PAIRS) {
      expect(t[fg], `${fg} missing`).toBeDefined()
      expect(t[bg], `${bg} missing`).toBeDefined()
      expect(contrast(t[fg], t[bg]), `${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5)
    }
  })

  it('computes contrast like the WCAG formula', () => {
    expect(contrast('#000000', '#ffffff')).toBeCloseTo(21, 5)
    expect(contrast('#c0392b', '#16181c')).toBeLessThan(4.5) // the dark-mode bug this guard exists for
  })
})
