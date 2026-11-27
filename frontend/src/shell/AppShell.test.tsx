import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import type { ChatsProps } from '../conversations/ChatList'
import type { Saved } from '../conversations/useConversation'
import type { Target } from './chatTarget'
import type { Page } from './route'
import { AppShell } from './AppShell'

const doc = { document_id: 'doc_1', title: 'a.md', connector: 'markdown', created_at: '2026-01-01' }
const documents = { documents: [doc], loaded: true, message: '', refresh: () => {}, remove: async () => {} }
const render = (
  page: Page,
  docs: Partial<typeof documents> = {},
  chat: Target = { key: 0, adopted: false },
  saved: Saved = { status: 'loading' },
  chats?: ChatsProps,
) =>
  renderToStaticMarkup(
    <AppShell
      page={page}
      collapsed={false}
      onToggleCollapse={() => {}}
      chat={chat}
      saved={saved}
      onConversation={() => {}}
      onNewChat={() => {}}
      chats={chats}
      documents={{ ...documents, ...docs }}
      quality={{ quality: null, message: '' }}
    />,
  )
const THREAD = 'aria-label="Ask a question about your documents"'

describe('AppShell', () => {
  it('shows the chat, not the Documents page, on the chat page', () => {
    const html = render('chat')

    expect(html).toContain(THREAD)
    expect(html).not.toContain('<div hidden=""')
    expect(html).not.toContain('>Documents</h1>')
  })

  it('keeps the chat mounted but hidden on the Documents page', () => {
    const html = render('documents')

    expect(html).toContain('<h1 class="mb-4 text-xl font-semibold">Documents</h1>')
    const hidden = html.indexOf('<div hidden=""')
    expect(hidden).toBeGreaterThanOrEqual(0)
    expect(html.indexOf(THREAD)).toBeGreaterThan(hidden)
    expect(html.indexOf(THREAD)).toBeLessThan(html.indexOf('>Documents</h1>'))
  })

  it('shows the Connectors page, with the chat mounted but hidden, on the connectors page only', () => {
    const html = render('connectors')

    expect(html).toContain('>Connectors and quality</h1>')
    expect(html.indexOf('<div hidden=""')).toBeGreaterThanOrEqual(0)
    expect(html.indexOf(THREAD)).toBeLessThan(html.indexOf('>Connectors and quality</h1>'))
    for (const page of ['chat', 'documents'] as const) expect(render(page)).not.toContain('>Connectors and quality</h1>')
  })

  it('hides the inline sidebar below md and offers a menu button there', () => {
    const html = render('chat')

    expect(html).toContain('class="hidden md:block"')
    expect(html).toContain('aria-label="Open menu"')
  })

  it('has no upload bar on any page but Documents', () => {
    expect(render('chat')).not.toContain('Upload a PDF or Word document')
    expect(render('documents')).toContain('Upload a PDF or Word document')
  })

  describe('first run', () => {
    const HEADLINE = 'Ask questions about your documents'
    const empty = { documents: [] }

    it('shows first-run, not the chat, with no documents, and the sidebar counts zero', () => {
      const html = render('chat', empty)
      expect(html).toContain(HEADLINE)
      expect(html).toContain('Documents (0)')
      expect(html).toContain('<div hidden=""')
    })

    it('shows nothing for the chat page until the list has loaded', () => {
      const html = render('chat', { ...empty, loaded: false })
      expect(html).not.toContain(HEADLINE)
      expect(html).not.toContain(THREAD)
    })

    it('shows the chat, not first-run, when documents exist; first-run again once all are gone', () => {
      expect(render('chat')).not.toContain(HEADLINE)
      expect(render('chat', empty)).toContain(HEADLINE)
    })

    it('shows the chat when the list failed to load', () => {
      const html = render('chat', { ...empty, message: "Couldn't load documents. Try again." })
      expect(html).not.toContain(HEADLINE)
      expect(html).toContain(THREAD)
    })

    it('does not show first-run on the Documents page', () => {
      expect(render('documents', empty)).not.toContain(HEADLINE)
    })
  })
})

describe('AppShell opening a saved chat', () => {
  const opening: Target = { key: 1, id: 'conv_A', adopted: false }
  const found: Saved = {
    status: 'found',
    conversation: { conversation_id: 'conv_A', title: 'Refunds', turns: [] },
  }

  it('shows the thread once the saved chat has loaded', () => {
    expect(render('chat', {}, opening, found)).toContain(THREAD)
  })

  it('shows no thread while it loads', () => {
    const html = render('chat', {}, opening, { status: 'loading' })

    expect(html).toContain('Loading chat')
    expect(html).not.toContain(THREAD)
  })

  it('says "Conversation not found" for an unknown id, with no thread', () => {
    const html = render('chat', {}, opening, { status: 'not-found' })

    expect(html).toContain('Conversation not found')
    expect(html).not.toContain(THREAD)
  })

  it("explains a failed load without the server's detail", () => {
    const html = render('chat', {}, opening, { status: 'error' })

    expect(html).toContain("Couldn&#x27;t load this chat. Try again.")
    expect(html).not.toContain(THREAD)
  })

  it('keeps the live thread of a new chat that was just saved', () => {
    const html = render('chat', {}, { key: 1, id: 'conv_A', adopted: true }, { status: 'loading' })

    expect(html).toContain(THREAD)
    expect(html).not.toContain('Loading chat')
  })

  it('shows an empty chat for New chat', () => {
    const html = render('chat', {}, { key: 2, adopted: false })

    expect(html).toContain(THREAD)
    expect(html).not.toContain('Conversation not found')
  })
})

describe('AppShell chat header', () => {
  const list = (title: string): ChatsProps => ({
    items: [{ conversation_id: 'conv_A', title, updated_at: '2026-11-02T10:00:00Z' }],
    currentId: 'conv_A',
    message: '',
    onRename: () => {},
    onDelete: () => {},
  })
  const open: Target = { key: 1, id: 'conv_A', adopted: false }
  const found: Saved = { status: 'found', conversation: { conversation_id: 'conv_A', title: 'Loaded title', turns: [] } }
  const header = (html: string) => html.match(/<header[^>]*>.*?<\/header>/)?.[0] ?? ''

  it('shows the open chat title from the list, and what the chat searches', () => {
    const h = header(render('chat', {}, open, found, list('Refund window')))

    expect(h).toMatch(/<h1[^>]*>Refund window<\/h1>/)
    expect(h).toContain('Searching all 1 document<')
  })

  it('follows a rename in the list', () => {
    expect(header(render('chat', {}, open, found, list('Returns')))).toContain('>Returns</h1>')
  })

  it("uses the loaded chat's own title until the list has it", () => {
    expect(header(render('chat', {}, open, found))).toContain('>Loaded title</h1>')
  })

  it('says "New chat" for an empty chat', () => {
    expect(header(render('chat'))).toContain('>New chat</h1>')
  })

  it('counts the indexed documents', () => {
    const two = [doc, { ...doc, document_id: 'doc_2' }]

    expect(header(render('chat', { documents: two }))).toContain('Searching all 2 documents')
  })

  it('has no header on the loading, not-found or error screens', () => {
    for (const status of ['loading', 'not-found', 'error'] as const) expect(render('chat', {}, open, { status })).not.toContain('<header')
  })

  it('sits inside the hidden chat wrapper on another page', () => {
    const html = render('documents')

    expect(html.indexOf('<div hidden=""')).toBeLessThan(html.indexOf('<header'))
    expect(html.indexOf('<header')).toBeLessThan(html.indexOf('>Documents</h1>'))
  })
})


