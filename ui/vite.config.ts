import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // Inject a build timestamp into index.html so every build
    // produces a unique HTML document.  Combined with the
    // Cache-Control: no-store middleware on the Python side,
    // this guarantees that the Pinokio proxy and browser
    // always fetch the latest bundle after a rebuild.
    {
      name: 'inject-build-stamp',
      transformIndexHtml(html) {
        return html.replace(
          '</head>',
          `<meta name="build-stamp" content="${Date.now()}" />\n  </head>`,
        )
      },
    },
  ],
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
