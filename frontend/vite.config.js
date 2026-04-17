import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/recommend": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/metrics": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/results": { target: "http://127.0.0.1:8000", changeOrigin: true }
    }
  }
});
