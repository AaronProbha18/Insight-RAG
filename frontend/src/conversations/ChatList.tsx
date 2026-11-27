import { PencilIcon, Trash2Icon } from 'lucide-react'
import { type FormEvent, useState } from 'react'
import { Button } from '@/components/ui/button'
import { isPlainClick, navigate } from '../shell/route'
import type { ConversationSummary } from './api'
import { saveAction } from './saveAction'

export type ChatsProps = {
  items: ConversationSummary[]
  /** The chat that is open, if any. */
  currentId?: string
  /** A failure from the last rename or delete, or ''. */
  message: string
  onRename: (id: string, title: string) => void
  onDelete: (id: string) => void
}

type Mode = 'idle' | 'rename' | 'delete'

type RowProps = {
  chat: ConversationSummary
  current: boolean
  mode: Mode
  onStartRename: () => void
  onStartDelete: () => void
  onCancel: () => void
  onSave: (title: string) => void
  onConfirmDelete: () => void
  onNavigate?: () => void
}

/** One chat: a link with Rename and Delete, or the inline rename form, or the delete confirmation. */
export function ChatRowView({
  chat,
  current,
  mode,
  onStartRename,
  onStartDelete,
  onCancel,
  onSave,
  onConfirmDelete,
  onNavigate,
}: RowProps) {
  const href = `/app/chats/${encodeURIComponent(chat.conversation_id)}`
  if (mode === 'rename') {
    const submit = (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault()
      onSave((e.currentTarget.elements.namedItem('title') as HTMLInputElement).value)
    }
    return (
      <li>
        <form onSubmit={submit} className="flex items-center gap-1">
          <input
            name="title"
            aria-label="Chat title"
            defaultValue={chat.title}
            maxLength={200}
            ref={(el) => el?.focus()}
            onKeyDown={(e) => e.key === 'Escape' && onCancel()}
            className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          />
          <Button type="submit" size="sm">
            Save
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        </form>
      </li>
    )
  }
  if (mode === 'delete') {
    return (
      <li>
        <div role="group" aria-label={`Delete ${chat.title}`} className="flex items-center gap-1 px-2 text-sm">
          <span className="min-w-0 flex-1">Delete this chat?</span>
          <Button type="button" variant="destructive" size="sm" onClick={onConfirmDelete}>
            Delete
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </li>
    )
  }
  return (
    <li className="flex items-center gap-1">
      <a
        href={href}
        aria-current={current ? 'page' : undefined}
        onClick={(e) => {
          if (isPlainClick(e)) {
            e.preventDefault()
            navigate(href)
          }
          onNavigate?.()
        }}
        className="min-w-0 flex-1 truncate rounded-md px-2 py-1.5 text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 aria-[current=page]:bg-muted aria-[current=page]:font-medium"
      >
        {chat.title}
      </a>
      <Button type="button" variant="ghost" size="icon-sm" aria-label={`Rename ${chat.title}`} onClick={onStartRename}>
        <PencilIcon aria-hidden="true" />
      </Button>
      <Button type="button" variant="ghost" size="icon-sm" aria-label={`Delete ${chat.title}`} onClick={onStartDelete}>
        <Trash2Icon aria-hidden="true" />
      </Button>
    </li>
  )
}

/** The Chats section of the sidebar. Renders nothing when there are no chats. */
export function ChatList({ chats, onNavigate }: { chats: ChatsProps; onNavigate?: () => void }) {
  const [editing, setEditing] = useState<{ id: string; mode: Mode }>()
  if (chats.items.length === 0) return null
  const close = () => setEditing(undefined)
  return (
    <section aria-labelledby="chats-heading" className="min-h-0 max-h-[40vh] overflow-y-auto">
      <h2 id="chats-heading" className="px-2 pb-1 text-xs font-medium text-muted-foreground">
        Chats
      </h2>
      <ul className="flex flex-col gap-0.5">
        {chats.items.map((chat) => {
          const id = chat.conversation_id
          return (
            <ChatRowView
              key={id}
              chat={chat}
              current={id === chats.currentId}
              mode={editing?.id === id ? editing.mode : 'idle'}
              onStartRename={() => setEditing({ id, mode: 'rename' })}
              onStartDelete={() => setEditing({ id, mode: 'delete' })}
              onCancel={close}
              onSave={(entered) => {
                const action = saveAction(chat.title, entered)
                if (action.kind === 'save') chats.onRename(id, action.title)
                close()
              }}
              onConfirmDelete={() => {
                chats.onDelete(id)
                close()
              }}
              onNavigate={onNavigate}
            />
          )
        })}
      </ul>
      {chats.message && (
        <p role="status" className="px-2 pt-1 text-xs text-muted-foreground">
          {chats.message}
        </p>
      )}
    </section>
  )
}
