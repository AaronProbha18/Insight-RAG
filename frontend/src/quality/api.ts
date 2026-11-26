export type Metrics = { correctness: number; faithfulness: number; abstention: number }

export type ConnectorQuality = {
  result: 'PASS' | 'FAIL'
  failed: string[]
  metrics: Metrics
  // Both or neither: /quality omits them for a summary written before the counts existed.
  answerable_questions?: number
  unanswerable_questions?: number
}

export type Quality =
  | { state: 'not_measured'; eval_command: string }
  | {
      state: 'measured'
      result: 'PASS' | 'FAIL'
      run_at: string
      models: { chat: string; embeddings: string; judge: string }
      threshold: number
      config_matches: boolean
      config_differences: ('chat' | 'embeddings' | 'threshold')[]
      connectors: Record<string, ConnectorQuality>
    }

/** GET /quality. Throws on any failure; the caller shows a generic message. */
export async function getQuality(): Promise<Quality> {
  const response = await fetch('/quality')
  if (!response.ok) throw new Error(`get quality failed (${response.status})`)
  return response.json()
}
