import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  resolve: {
    alias: [
      {
        find: /lucide-react\/dist\/esm\/icons\/fingerprint\.js$/,
        replacement: 'lucide-react/dist/esm/icons/shield.js',
      },
    ],
  },
});
