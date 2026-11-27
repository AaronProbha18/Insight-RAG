import { useState } from 'react'
import { useConversation } from './conversations/useConversation'
import { useConversations } from './conversations/useConversations'
import { useDocuments } from './documents/useDocuments'
import { useQuality } from './quality/useQuality'
import { useUnanswered } from './unanswered/useUnanswered'
import { AppShell } from './shell/AppShell'
import { useCollapsed } from './shell/collapsed'
import { adopt, leavesAfterDelete, newChat, syncTarget, type Target } from './shell/chatTarget'
import { navigate, replace, useChatId, useRoute } from './shell/route'

export default function App() {
  const page = useRoute()
  const chatId = useChatId()
  const [collapsed, toggleCollapsed] = useCollapsed()
  const documents = useDocuments()
  const quality = useQuality()
  const conversations = useConversations()
  const unanswered = useUnanswered(30)
  // What the chat shows. It follows the URL only while the chat page is open, so the chat
  // survives a visit to Documents; a new `key` remounts it.
  const [target, setTarget] = useState<Target>(() => syncTarget({ key: 0, adopted: false }, chatId))
  const next = page === 'chat' ? syncTarget(target, chatId) : target
  if (next !== target) setTarget(next)
  const saved = useConversation(next.id && !next.adopted ? next.id : undefined)
  const startNewChat = () => {
    navigate('/')
    setTarget((t) => newChat(t))
  }

  return (
    <AppShell
      page={page}
      collapsed={collapsed}
      onToggleCollapse={toggleCollapsed}
      chat={next}
      saved={saved}
      onConversation={(id) => {
        setTarget((t) => adopt(t, id))
        replace(`/app/chats/${encodeURIComponent(id)}`)
        conversations.refresh()
      }}
      onNewChat={startNewChat}
      chats={{
        items: conversations.conversations,
        currentId: page === 'chat' ? next.id : undefined,
        message: conversations.message,
        onRename: (id, title) => void conversations.rename(id, title),
        onDelete: async (id) => {
          if (leavesAfterDelete(await conversations.remove(id), id, next.id)) startNewChat()
        },
      }}
      unansweredCount={unanswered.items.length}
      onAnswer={unanswered.refresh}
      documents={documents}
      quality={quality}
    />
  )
}
