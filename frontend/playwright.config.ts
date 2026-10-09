// Browser smoke tests: the REAL backend (fresh SQLite database, mock agent
// only, no API key) behind the PRODUCTION build served by `vite preview`.
//
//   npm run build && npm run e2e
//
// Needs a Chromium for Playwright: `npx playwright install chromium` (or set
// PLAYWRIGHT_CHROMIUM to an existing Chromium binary). Ports 8011 and 4174 are
// used so a running dev setup (8000/5173) is not disturbed.
import { defineConfig } from "@playwright/test";

const BACKEND_PORT = 8011;
const UI_PORT = 4174;

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${UI_PORT}`,
    viewport: { width: 1360, height: 900 },
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM }
      : {},
  },
  webServer: [
    {
      command:
        "rm -rf e2e/.tmp && mkdir -p e2e/.tmp && cd ../backend && " +
        `DATABASE_URL=sqlite:///../frontend/e2e/.tmp/e2e.db GEMINI_API_KEY= ` +
        `.venv/bin/uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port ${BACKEND_PORT}`,
      url: `http://127.0.0.1:${BACKEND_PORT}/health`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `vite preview --port ${UI_PORT} --strictPort`,
      url: `http://127.0.0.1:${UI_PORT}`,
      env: { AGENTFLOW_BACKEND: `http://127.0.0.1:${BACKEND_PORT}` },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
