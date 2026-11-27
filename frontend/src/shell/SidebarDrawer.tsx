import { Dialog } from '@base-ui/react/dialog'
import { MenuIcon } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import type { ChatsProps } from '../conversations/ChatList'
import type { Page } from './route'
import { Sidebar } from './Sidebar'

type Props = { page: Page; documentCount: number; unansweredCount: number; onNewChat: () => void; chats?: ChatsProps }

/** Below md: a menu button that opens the sidebar as a drawer. Both are hidden from md up. */
export function SidebarDrawer({ page, documentCount, unansweredCount, onNewChat, chats }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger render={<Button variant="ghost" size="icon" aria-label="Open menu" className="md:hidden" />}>
        <MenuIcon aria-hidden="true" />
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 bg-foreground/40 md:hidden" />
        <Dialog.Popup className="fixed inset-y-0 left-0 text-foreground shadow-lg md:hidden">
          <Dialog.Title className="sr-only">Main menu</Dialog.Title>
          <Sidebar
            page={page}
            documentCount={documentCount}
            unansweredCount={unansweredCount}
            collapsed={false}
            onNewChat={onNewChat}
            chats={chats}
            onNavigate={() => setOpen(false)}
          />
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
