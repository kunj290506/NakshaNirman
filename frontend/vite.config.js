import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://localhost:8001',
                changeOrigin: true,
                ws: true,
                rewrite: (path) => path,
                timeout: 60000,
                onError: (err, req, res) => {
                    console.error('Proxy error:', err)
                },
            },
        },
    },
})
