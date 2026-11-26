/// <reference types="node" /> // this test reads index.css from disk
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// Guards the styling rules in docs/guides/frontend-development-guide.md §6.
const css = readFileSync(fileURLToPath(new URL('./index.css', import.meta.url)), 'utf-8')
const COLOUR = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(/

/**
 * Text-on-background pairs the stylesheet and the kit components use.
 * Semi-transparent backgrounds (e.g. `bg-muted/40`) are not measured: their
 * contrast depends on what is painted behind them.
 */
const PAIRS: [text: string, background: string][] = [
  ['--foreground', '--background'], // page text, Connectors page and cards
  ['--foreground', '--muted'], // assistant messages, user bubble, current sidebar item, metric bar fill and 0.80 mark on their track
  ['--muted-foreground', '--background'], // secondary text, card extensions and run line
  ['--muted-foreground', '--muted'], // secondary text on muted surfaces
  ['--primary-foreground', '--primary'], // primary buttons
  ['--secondary-foreground', '--secondary'], // secondary buttons
  ['--accent-foreground', '--accent'], // hovered items
  ['--popover-foreground', '--popover'], // menus
  ['--background', '--foreground'], // tooltips
  ['--destructive', '--background'], // errors, Below bar, not-current note
]

function tokenBlocks(source: string): { light: string; dark: string; rest: string } {
  const light = /^:root\s*{([^}]*)}/m.exec(source)
  const dark = /@media \(prefers-color-scheme: dark\)\s*{\s*:root\s*{([^}]*)}\s*}/.exec(source)
  if (!light || !dark) throw new Error('index.css must define :root tokens and a dark-mode :root block')
  return { light: light[1], dark: dark[1], rest: source.replace(light[0], '').replace(dark[0], '') }
}

/** `--name: oklch(L C H)` -> [L, C, H]. Tokens with alpha are skipped: see PAIRS. */
function tokens(block: string): Record<string, [number, number, number]> {
  return Object.fromEntries(
    [...block.matchAll(/(--[\w-]+):\s*oklch\(([\d.]+)\s+([\d.]+)\s+([\d.]+)\)/g)].map((m) => [
      m[1],
      [Number(m[2]), Number(m[3]), Number(m[4])],
    ]),
  )
}

/** WCAG relative luminance of an OKLCH colour, via OKLab -> linear sRGB (Ottosson). */
export function luminance([L, C, H]: [number, number, number]): number {
  const a = C * Math.cos((H * Math.PI) / 180)
  const b = C * Math.sin((H * Math.PI) / 180)
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  const clamp = (x: number) => Math.min(1, Math.max(0, x))
  const r = clamp(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)
  const g = clamp(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)
  const bl = clamp(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)
  return 0.2126 * r + 0.7152 * g + 0.0722 * bl
}

export function contrast(a: [number, number, number], b: [number, number, number]): number {
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
    expect(luminance([1, 0, 0])).toBeCloseTo(1, 5)
    expect(contrast([0, 0, 0], [1, 0, 0])).toBeCloseTo(21, 5)
    // the shadcn default this guard caught: light muted-foreground on muted
    expect(contrast([0.556, 0, 0], [0.97, 0, 0])).toBeLessThan(4.5)
  })
})
