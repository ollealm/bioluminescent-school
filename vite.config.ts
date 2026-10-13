import { defineConfig } from "vite";

export default defineConfig({
  build: {
    // The largest chunk is three.js for the CPU fallback (~500 kB), loaded
    // only on devices without WebGPU; the GPU path's entry is ~50 kB.
    chunkSizeWarningLimit: 600,
  },
});
