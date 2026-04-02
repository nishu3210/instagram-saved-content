import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5001",
      "/static": "http://127.0.0.1:5001",
      "/legacy": "http://127.0.0.1:5001"
    }
  },
  build: {
    outDir: "../static/app",
    emptyOutDir: true
  },
  base: "/static/app/"
});
