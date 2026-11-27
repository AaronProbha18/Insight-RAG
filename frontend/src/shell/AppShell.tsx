import { useState } from 'react'
import { ChatHeader } from '../chat/ChatHeader'
import { ChatPane } from '../chat/ChatPane'
import { FirstRun } from '../chat/FirstRun'
import { DocumentsPage } from '../documents/DocumentsPage'
import type { useDocuments } from '../documents/useDocuments'
import { ConnectorsPage } from '../connectors/ConnectorsPage'
import type { ChatsProps } from '../conversations/ChatList'
import type { Saved } from '../conversations/useConversation'
import { UnansweredPage } from '../unanswered/UnansweredPage'
import type { useQuality } from '../quality/useQuality'
import type { Target } from './chatTarget'
import type { Page } from './route'
import { Sidebar } from './Sidebar'
import { SidebarDrawer } from './SidebarDrawer'

type Props = {
  page: Page
  collapsed: boolean
  onToggleCollapse: () => void
  chat: Target
  saved: Saved // the saved chat being opened; unused for a new or just-saved chat
  onConversation: (id: string) => void
  onNewChat: () => void
  chats?: ChatsProps
  documents: ReturnType<typeof useDocuments>
  quality: ReturnType<typeof useQuality>
}

/** Sidebar plus one page. The chat stays mounted (hidden) while Documents shows, so it survives navigation. */
export function AppShell({
  page,
  collapsed,
  onToggleCollapse,
  chat,
  saved,
  onConversation,
  onNewChat,
  chats,
  documents,
  quality,
}: Props) {
  const count = documents.documents.length
  const [samples, setSamples] = useState(false)
  const firstRun = documents.loaded && count === 0 && !documents.message
  const opening = chat.id !== undefined && !chat.adopted // a saved chat is being opened
  // The list is the source for the title, so a rename shows here as soon as it refreshes.
  const title =
    chats?.items.find((c) => c.conversation_id === chat.id)?.title ??
    (saved.status === 'found' ? saved.conversation.title : 'New chat')
  return (
    <div className="flex h-dvh">
      <div className="hidden md:block">
        <Sidebar
          page={page}
          documentCount={count}
          collapsed={collapsed}
          onToggleCollapse={onToggleCollapse}
          onNewChat={onNewChat}
          chats={chats}
        />
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-border p-1 md:hidden">
          <SidebarDrawer page={page} documentCount={count} onNewChat={onNewChat} chats={chats} />
        </div>
        {documents.loaded && (
          <div hidden={page !== 'chat' || firstRun} className="min-h-0 flex-1">
            {opening && saved.status === 'loading' && <p role="status" className="p-6 text-sm text-muted-foreground">Loading chat</p>}
            {opening && saved.status === 'not-found' && <p className="p-6 text-sm">Conversation not found</p>}
            {opening && saved.status === 'error' && <p className="p-6 text-sm">Couldn't load this chat. Try again.</p>}
            {(!opening || saved.status === 'found') && (
              <div className="flex h-full flex-col">
                <ChatHeader title={title} documentCount={count} />
                <div className="min-h-0 flex-1">
                  <ChatPane
                    key={chat.key}
                    samples={samples}
                    conversationId={saved.status === 'found' ? saved.conversation.conversation_id : undefined}
                    turns={saved.status === 'found' ? saved.conversation.turns : undefined}
                    onConversation={onConversation}
                  />
                </div>
              </div>
            )}
          </div>
        )}
        {page === 'chat' && firstRun && (
          <div className="min-h-0 flex-1">
            <FirstRun
              onIndexed={() => {
                setSamples(true)
                documents.refresh()
              }}
            />
          </div>
        )}
        {page === 'documents' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <DocumentsPage {...documents} />
          </div>
        )}
        {page === 'connectors' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <ConnectorsPage quality={quality.quality} qualityMessage={quality.message} documents={documents.documents} />
          </div>
        )}
        {page === 'unanswered' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <UnansweredPage />
          </div>
        )}
      </div>
    </div>
  )
}
