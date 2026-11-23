import type { Citation } from './adapter'

/** Sources for one answer; each expands to the cited text. Renders nothing when empty. */
export function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null
  return (
    <section className="mt-3 border-t border-border pt-2 text-sm" aria-label="Sources">
      <ol className="list-decimal space-y-1 ps-5">
        {citations.map((c) => (
          <li key={c.id}>
            <details>
              <summary className="cursor-pointer rounded-sm text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                [{c.id}] {c.title} — {c.locator}
              </summary>
              <blockquote className="mt-1 mb-2 border-s-2 border-border ps-3 text-foreground">{c.text}</blockquote>
            </details>
          </li>
        ))}
      </ol>
    </section>
  )
}
