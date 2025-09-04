import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    lib: {
      entry: 'src/ParticleSystem.jsx',
      name: 'ParticleSystem',
      fileName: 'particle-system',
      formats: ['es']
    },
    rollupOptions: {
      external: ['react', 'react-dom'],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM'
        },
        manualChunks: undefined
      },
      treeshake: {
        moduleSideEffects: false
      }
    },
    minify: true,
    target: 'es2020'
  },
  resolve: {
    alias: {
      // Force three.js to use ES modules for better tree-shaking
      'three': 'three/src/Three.js'
    }
  },
  define: {
    // Remove development code
    'process.env.NODE_ENV': '"production"'
  }
})