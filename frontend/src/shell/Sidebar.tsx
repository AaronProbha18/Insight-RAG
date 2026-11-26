import {
  BookOpenIcon,
  FileTextIcon,
  MessageSquarePlusIcon,
  PanelLeftCloseIcon,
  PanelLeftOpenIcon,
  PlugIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Quality } from '../quality/Quality'
import { isPlainClick, navigate, type Page } from './route'

const DOCS_URL = 'https://github.com/AaronProbha18/Plumbline/blob/main/docs/architecture/overview.md'
const item =
  'flex items-center gap-2 rounded-md px-2 py-1.5 text-sm outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50'

type Props = {
  page: Page
  documentCount: number
  collapsed: boolean
  onToggleCollapse: () => void
  onNewChat: () => void
}

/** Main navigation. Collapsed shows icons only; every item keeps its name as screen-reader text. */
export function Sidebar({ page, documentCount, collapsed, onToggleCollapse, onNewChat }: Props) {
  const label = collapsed ? 'sr-only' : undefined
  const group = `px-2 pb-1 text-xs font-medium text-muted-foreground ${collapsed ? 'sr-only' : ''}`
  const CollapseIcon = collapsed ? PanelLeftOpenIcon : PanelLeftCloseIcon
  return (
    <nav
      aria-label="Main"
      className={`flex h-full flex-col gap-4 border-r border-border bg-background p-2 ${collapsed ? 'w-14' : 'w-60'}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 px-2 font-semibold">
          <img src="/favicon.svg" alt="" className="size-5" />
          <span className={label}>Plumbline</span>
        </span>
        <Button
          variant="ghost"
          size="icon"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
          onClick={onToggleCollapse}
        >
          <CollapseIcon aria-hidden="true" />
        </Button>
      </div>

      <button type="button" onClick={onNewChat} className={`${item} font-medium`}>
        <MessageSquarePlusIcon aria-hidden="true" className="size-4 shrink-0" />
        <span className={label}>New chat</span>
      </button>

      <div>
        <p className={group}>Knowledge</p>
        <a
          href="/app/documents"
          aria-current={page === 'documents' ? 'page' : undefined}
          onClick={(e) => {
            if (!isPlainClick(e)) return
            e.preventDefault()
            navigate('/app/documents')
          }}
          className={`${item} aria-[current=page]:bg-muted aria-[current=page]:font-medium`}
        >
          <FileTextIcon aria-hidden="true" className="size-4 shrink-0" />
          <span className={label}>Documents ({documentCount})</span>
        </a>
        <span aria-disabled="true" className={`${item} cursor-not-allowed text-muted-foreground hover:bg-transparent`}>
          <PlugIcon aria-hidden="true" className="size-4 shrink-0" />
          <span className={label}>Connectors (Coming soon)</span>
        </span>
      </div>

      <div>
        <p className={group}>Insights</p>
        <Quality compact={collapsed} />
      </div>

      <a href={DOCS_URL} target="_blank" rel="noreferrer" className={`${item} mt-auto`}>
        <BookOpenIcon aria-hidden="true" className="size-4 shrink-0" />
        <span className={label}>Docs</span>
      </a>
    </nav>
  )
}
