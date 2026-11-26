import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

// Dev only: forward API calls to the backend. In production FastAPI serves this UI.
const backend = process.env.VITE_BACKEND ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.join(import.meta.dirname, 'src') } },
  server: { proxy: { '/chat': backend, '/connectors': backend, '/documents': backend, '/feedback': backend, '/health': backend, '/quality': backend } },
})
