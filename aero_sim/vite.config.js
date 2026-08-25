import { defineConfig } from 'vite'
import cesium from 'vite-plugin-cesium'

export default defineConfig({
  plugins: [cesium()],
  server: {
    open: true,
    port: 5173,
  },
  build: {
    outDir: 'dist',
  },
})
