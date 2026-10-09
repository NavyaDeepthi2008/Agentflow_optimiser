/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { configDefaults } from "vitest/config";

// In development the dashboard calls "/api/..." on its own origin and Vite
// forwards it to the FastAPI backend, so the browser never needs CORS.
// (The backend also allows the local dev origins via CORS_ORIGINS, for setups
// that call it directly with VITE_API_BASE_URL.)
const backend = process.env.AGENTFLOW_BACKEND ?? "http://127.0.0.1:8000";
const proxy = {
  "/api": {
    target: backend,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, proxy },
  preview: { host: "127.0.0.1", port: 4173, proxy },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
    exclude: [...configDefaults.exclude, "e2e/**"], // browser tests run with Playwright
  },
});
