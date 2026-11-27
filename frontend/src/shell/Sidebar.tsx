import {
  BookOpenIcon,
  CircleHelpIcon,
  FileTextIcon,
  MessageSquarePlusIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlugIcon,
} from 'lucide-react'
import type { ReactElement } from 'react'
import { Button } from '@/components/ui/button'
import { ChatList, type ChatsProps } from '../conversations/ChatList'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { isPlainClick, navigate, type Page } from './route'

const DOCS_URL = 'https://plumbline-docs.netlify.app/'
const item =
  'flex items-center gap-2 rounded-md px-2 text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50'

type Props = {
  page: Page
  documentCount: number
  unansweredCount: number // groups in the last 30 days
  collapsed: boolean
  onToggleCollapse?: () => void // omitted in the drawer, where collapsing means nothing
  onNewChat: () => void
  onNavigate?: () => void // after a navigating click, so the drawer can close
  chats?: ChatsProps // the saved chats; the section is hidden when there are none or the sidebar is collapsed
}

/** A tooltip on the right of `children`, shown only when `show` (the collapsed, icon-only sidebar). */
function Tip({ label, show, children }: { label: string; show: boolean; children: ReactElement }) {
  if (!show) return children
  return (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  )
}

/** Main navigation. Collapsed shows icons only; every item keeps its name as screen-reader text. */
export function Sidebar({ page, documentCount, unansweredCount, collapsed, onToggleCollapse, onNewChat, onNavigate, chats }: Props) {
  const label = collapsed ? 'sr-only' : undefined
  // Collapsed: icons centred and spaced apart.
  const row = collapsed ? `${item} justify-center py-2.5` : `${item} py-1.5`
  const stack = collapsed ? 'flex flex-col gap-2' : undefined
  const group = `px-2 pb-1 text-xs font-medium text-muted-foreground ${collapsed ? 'sr-only' : ''}`
  // A link to an app page. A function, not a component, so the anchor stays in Sidebar's own element tree.
  const pageLink = (href: string, current: boolean, Icon: typeof FileTextIcon, text: string, tip = text) => (
    <Tip label={tip} show={collapsed}>
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
        className={`${row} aria-[current=page]:bg-muted aria-[current=page]:font-medium`}
      >
        <Icon aria-hidden="true" className="size-4 shrink-0" />
        <span className={label}>{text}</span>
      </a>
    </Tip>
  )
  const CollapseIcon = collapsed ? PanelLeftOpenIcon : PanelLeftCloseIcon
  return (
    <TooltipProvider delay={200}>
      <nav
        aria-label="Main"
        className={`flex h-full flex-col border-r border-border bg-background p-2 ${collapsed ? 'w-14 gap-6' : 'w-60 gap-4'}`}
      >
        {/* Collapsed: the expand button sits over the logo and shows on hover or focus. It stays
            focusable (opacity, not display), so keyboard users reach it. */}
        <div className={`group/head relative flex items-center gap-2 ${collapsed ? 'h-8 justify-center' : 'justify-between'}`}>
          <span
            className={`flex items-center gap-2 px-2 font-semibold ${collapsed ? 'group-hover/head:opacity-0 group-has-[:focus-visible]/head:opacity-0' : ''}`}
          >
            <img src="/favicon.svg" alt="" className="size-5 shrink-0" />
            <span className={label}>Plumbline</span>
          </span>
          {onToggleCollapse && (
            <Tip label="Expand sidebar" show={collapsed}>
              <Button
                variant="ghost"
                size="icon"
                aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                aria-expanded={!collapsed}
                onClick={onToggleCollapse}
                className={collapsed ? 'absolute inset-0 m-auto opacity-0 group-hover/head:opacity-100 group-has-[:focus-visible]/head:opacity-100' : undefined}
              >
                <CollapseIcon aria-hidden="true" />
              </Button>
            </Tip>
          )}
        </div>

        <Tip label="New chat" show={collapsed}>
          <button
            type="button"
            onClick={() => {
              onNewChat()
              onNavigate?.()
            }}
            className={`${row} font-medium`}
          >
            <MessageSquarePlusIcon aria-hidden="true" className="size-4 shrink-0" />
            <span className={label}>New chat</span>
          </button>
        </Tip>

        {chats && !collapsed && <ChatList chats={chats} onNavigate={onNavigate} />}

        <div className={stack}>
          <p className={group}>Knowledge</p>
          {pageLink('/app/documents', page === 'documents', FileTextIcon, `Documents (${documentCount})`)}
          {pageLink('/app/unanswered', page === 'unanswered', CircleHelpIcon, `Unanswered (${unansweredCount})`)}
          {pageLink('/app/connectors', page === 'connectors', PlugIcon, 'Connectors and quality')}
        </div>

        <Tip label="Docs" show={collapsed}>
          <a href={DOCS_URL} target="_blank" rel="noreferrer" className={`${row} mt-auto`}>
            <BookOpenIcon aria-hidden="true" className="size-4 shrink-0" />
            <span className={label}>Docs</span>
          </a>
        </Tip>
      </nav>
    </TooltipProvider>
  )
}
