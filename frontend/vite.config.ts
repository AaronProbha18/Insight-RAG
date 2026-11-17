import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev only: forward API calls to the backend. In production FastAPI serves this UI.
const backend = process.env.VITE_BACKEND ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/chat': backend, '/documents': backend, '/health': backend } },
})
