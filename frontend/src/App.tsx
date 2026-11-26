import { useState } from 'react'
import { useDocuments } from './documents/useDocuments'
import { useQuality } from './quality/useQuality'
import { AppShell } from './shell/AppShell'
import { useCollapsed } from './shell/collapsed'
import { navigate, useRoute } from './shell/route'

export default function App() {
  const page = useRoute()
  const [collapsed, toggleCollapsed] = useCollapsed()
  const documents = useDocuments()
  const quality = useQuality()
  const [chatKey, setChatKey] = useState(0) // a new key remounts the chat with an empty thread

  return (
    <AppShell
      page={page}
      collapsed={collapsed}
      onToggleCollapse={toggleCollapsed}
      chatKey={chatKey}
      onNewChat={() => {
        navigate('/')
        setChatKey((k) => k + 1)
      }}
      documents={documents}
      quality={quality}
    />
  )
}
