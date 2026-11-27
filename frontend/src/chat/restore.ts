import type { ThreadMessageLike } from '@assistant-ui/react'
import type { Turn } from '@/conversations/api'

/** A saved conversation as initial thread messages: the question, then the recorded answer. */
export function turnsToMessages(turns: Turn[]): ThreadMessageLike[] {
  return turns.flatMap((t) => [
    { role: 'user' as const, content: [{ type: 'text' as const, text: t.question }] },
    {
      role: 'assistant' as const,
      content: [{ type: 'text' as const, text: t.answer }],
      status: { type: 'complete' as const, reason: 'stop' as const },
      metadata: { custom: { citations: t.citations, abstained: t.abstained, answerId: t.answer_id } },
    },
  ])
}
