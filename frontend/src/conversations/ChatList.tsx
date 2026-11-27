import { CheckIcon, MessageCircleIcon, PencilIcon, Trash2Icon, XIcon } from 'lucide-react'
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
        <form onSubmit={submit} className="flex items-center gap-1 px-2">
          <MessageCircleIcon aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" />
          <input
            name="title"
            aria-label="Chat title"
            defaultValue={chat.title}
            maxLength={200}
            ref={(el) => el?.focus()}
            onFocus={(e) => e.currentTarget.select()}
            onKeyDown={(e) => e.key === 'Escape' && onCancel()}
            className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
          />
          <Button type="submit" variant="ghost" size="icon-sm" aria-label="Save title">
            <CheckIcon aria-hidden="true" />
          </Button>
          <Button type="button" variant="ghost" size="icon-sm" aria-label="Cancel rename" onClick={onCancel}>
            <XIcon aria-hidden="true" />
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
    <li className="group/row relative">
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
        className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 aria-[current=page]:bg-muted aria-[current=page]:font-medium"
      >
        <MessageCircleIcon aria-hidden="true" className="size-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate">{chat.title}</span>
      </a>
      {/* Shown on hover and on keyboard focus (opacity, not display, so they stay focusable), and always
          on touch screens, which have no hover. */}
      <div className="absolute inset-y-0 right-1 flex items-center gap-0.5 rounded-md bg-muted opacity-0 transition-opacity group-focus-within/row:opacity-100 group-hover/row:opacity-100 pointer-coarse:opacity-100">
        <Button type="button" variant="ghost" size="icon-sm" aria-label={`Rename ${chat.title}`} onClick={onStartRename}>
          <PencilIcon aria-hidden="true" />
        </Button>
        <Button type="button" variant="ghost" size="icon-sm" aria-label={`Delete ${chat.title}`} onClick={onStartDelete}>
          <Trash2Icon aria-hidden="true" />
        </Button>
      </div>
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
