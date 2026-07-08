import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:7860',
      '/classic': 'http://127.0.0.1:7860',
    },
  },
  // Strip console.* and debugger statements from the production bundle.
  // Dev mode (npm run dev) is unaffected — esbuild `drop` only runs at
  // build time.
  esbuild: {
    drop: ['console', 'debugger'],
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
