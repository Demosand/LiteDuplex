import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' — Electron открывает dist/index.html через file://
export default defineConfig({ base: './', plugins: [react()], build: { outDir: 'dist', emptyOutDir: true } });
