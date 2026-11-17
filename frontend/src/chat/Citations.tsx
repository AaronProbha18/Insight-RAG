import type { Citation } from './adapter'

/** Sources for one answer; each expands to the cited text. Renders nothing when empty. */
export function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null
  return (
    <section className="sources" aria-label="Sources">
      <ol>
        {citations.map((c) => (
          <li key={c.id}>
            <details>
              <summary>
                [{c.id}] {c.title} — {c.locator}
              </summary>
              <blockquote>{c.text}</blockquote>
            </details>
          </li>
        ))}
      </ol>
    </section>
  )
}
