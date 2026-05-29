/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
    globals: true,
  },
  server: {
    proxy: {
      '/auth': 'http://logic-backend:8080',
      '/experiments': 'http://logic-backend:8080',
      '/compare': 'http://logic-backend:8080',
    },
  },
})
