import type { Feedback } from './api'

type Reason = NonNullable<Feedback['reason']>

export const REASONS: { value: Reason; label: string }[] = [
  { value: 'wrong', label: 'Wrong' },
  { value: 'incomplete', label: 'Incomplete' },
  { value: 'wrong_source', label: 'Wrong source' },
  { value: 'should_have_answered', label: 'Should have answered' },
  { value: 'other', label: 'Other' },
]
