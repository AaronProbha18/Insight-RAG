import {
  BookOpenIcon,
  FileTextIcon,
  MessageSquarePlusIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlugIcon,
} from 'lucide-react'
import type { ReactElement } from 'react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Quality } from '../quality/Quality'
import { isPlainClick, navigate, type Page } from './route'

const DOCS_URL = 'https://github.com/AaronProbha18/Plumbline/blob/main/docs/architecture/overview.md'
const item =
  'flex items-center gap-2 rounded-md px-2 text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50'

type Props = {
  page: Page
  documentCount: number
  collapsed: boolean
  onToggleCollapse?: () => void // omitted in the drawer, where collapsing means nothing
  onNewChat: () => void
  onNavigate?: () => void // after a navigating click, so the drawer can close
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
export function Sidebar({ page, documentCount, collapsed, onToggleCollapse, onNewChat, onNavigate }: Props) {
  const label = collapsed ? 'sr-only' : undefined
  // Collapsed: icons centred and spaced apart.
  const row = collapsed ? `${item} justify-center py-2.5` : `${item} py-1.5`
  const stack = collapsed ? 'flex flex-col gap-2' : undefined
  const group = `px-2 pb-1 text-xs font-medium text-muted-foreground ${collapsed ? 'sr-only' : ''}`
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

      <div className={stack}>
        <p className={group}>Knowledge</p>
        <Tip label={`Documents (${documentCount})`} show={collapsed}>
        <a
          href="/app/documents"
          aria-current={page === 'documents' ? 'page' : undefined}
          onClick={(e) => {
            if (isPlainClick(e)) {
              e.preventDefault()
              navigate('/app/documents')
            }
            onNavigate?.()
          }}
          className={`${row} aria-[current=page]:bg-muted aria-[current=page]:font-medium`}
        >
          <FileTextIcon aria-hidden="true" className="size-4 shrink-0" />
          <span className={label}>Documents ({documentCount})</span>
        </a>
        </Tip>
        <Tip label="Connectors (Coming soon)" show={collapsed}>
          <span aria-disabled="true" className={`${row} cursor-not-allowed text-muted-foreground hover:bg-transparent`}>
            <PlugIcon aria-hidden="true" className="size-4 shrink-0" />
            <span className={label}>Connectors (Coming soon)</span>
          </span>
        </Tip>
      </div>

      <div className={stack}>
        <p className={group}>Insights</p>
        <Tip label="Quality" show={collapsed}>
          <div>
            <Quality compact={collapsed} itemClassName={`${row} w-full`} />
          </div>
        </Tip>
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
