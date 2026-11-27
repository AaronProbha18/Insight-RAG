import { useState } from 'react'
import { ChatPane } from '../chat/ChatPane'
import { FirstRun } from '../chat/FirstRun'
import { DocumentsPage } from '../documents/DocumentsPage'
import type { useDocuments } from '../documents/useDocuments'
import { ConnectorsPage } from '../connectors/ConnectorsPage'
import type { useQuality } from '../quality/useQuality'
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
  quality: ReturnType<typeof useQuality>
}

/** Sidebar plus one page. The chat stays mounted (hidden) while Documents shows, so it survives navigation. */
export function AppShell({ page, collapsed, onToggleCollapse, chatKey, onNewChat, documents, quality }: Props) {
  const count = documents.documents.length
  const [samples, setSamples] = useState(false)
  const firstRun = documents.loaded && count === 0 && !documents.message
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
        {documents.loaded && (
          <div hidden={page !== 'chat' || firstRun} className="min-h-0 flex-1">
            <ChatPane key={chatKey} samples={samples} />
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
      </div>
    </div>
  )
}
