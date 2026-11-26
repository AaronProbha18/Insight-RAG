import { ChatPane } from '../chat/ChatPane'
import { DocumentsPage } from '../documents/DocumentsPage'
import type { useDocuments } from '../documents/useDocuments'
import type { Page } from './route'
import { Sidebar } from './Sidebar'
import { SidebarDrawer } from './SidebarDrawer'

type Props = {
  page: Page
  collapsed: boolean
  onToggleCollapse: () => void
  chatKey: number
  onNewChat: () => void
  documents: ReturnType<typeof useDocuments>
}

/** Sidebar plus one page. The chat stays mounted (hidden) while Documents shows, so it survives navigation. */
export function AppShell({ page, collapsed, onToggleCollapse, chatKey, onNewChat, documents }: Props) {
  const count = documents.documents.length
  return (
    <div className="flex h-dvh">
      <div className="hidden md:block">
        <Sidebar
          page={page}
          documentCount={count}
          collapsed={collapsed}
          onToggleCollapse={onToggleCollapse}
          onNewChat={onNewChat}
        />
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-border p-1 md:hidden">
          <SidebarDrawer page={page} documentCount={count} onNewChat={onNewChat} />
        </div>
        <div hidden={page !== 'chat'} className="min-h-0 flex-1">
          <ChatPane key={chatKey} />
        </div>
        {page === 'documents' && (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <DocumentsPage {...documents} />
          </div>
        )}
      </div>
    </div>
  )
}
