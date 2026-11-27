import { useState } from 'react'
import { useConversation } from './conversations/useConversation'
import { useConversations } from './conversations/useConversations'
import { useDocuments } from './documents/useDocuments'
import { useQuality } from './quality/useQuality'
import { AppShell } from './shell/AppShell'
import { useCollapsed } from './shell/collapsed'
import { adopt, newChat, syncTarget, type Target } from './shell/chatTarget'
import { navigate, replace, useChatId, useRoute } from './shell/route'

export default function App() {
  const page = useRoute()
  const chatId = useChatId()
  const [collapsed, toggleCollapsed] = useCollapsed()
  const documents = useDocuments()
  const quality = useQuality()
  const conversations = useConversations()
  // What the chat shows. It follows the URL only while the chat page is open, so the chat
  // survives a visit to Documents; a new `key` remounts it.
  const [target, setTarget] = useState<Target>(() => syncTarget({ key: 0, adopted: false }, chatId))
  const next = page === 'chat' ? syncTarget(target, chatId) : target
  if (next !== target) setTarget(next)
  const saved = useConversation(next.id && !next.adopted ? next.id : undefined)

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
      onNewChat={() => {
        navigate('/')
        setTarget((t) => newChat(t))
      }}
      documents={documents}
      quality={quality}
    />
  )
}
