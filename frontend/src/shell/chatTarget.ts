/** What the mounted chat shows. A new `key` remounts it; `adopted` is a new chat that just got its id. */
export type Target = { key: number; id?: string; adopted: boolean }

/** Follows the URL: another chat id (or none) means a fresh pane; the same id keeps the pane as it is. */
export function syncTarget(t: Target, chatId: string | undefined): Target {
  return chatId === t.id ? t : { key: t.key + 1, id: chatId, adopted: false }
}

/** New chat: an empty thread. */
export const newChat = (t: Target): Target => ({ key: t.key + 1, adopted: false })

/** A new chat was saved under `id`. Same `key`, so its live thread stays while the URL changes. */
export const adopt = (t: Target, id: string): Target => ({ ...t, id, adopted: true })
