// Test helpers: a fake fetch that answers from recorded backend responses
// (src/test/fixtures, produced by backend/scripts/export_frontend_fixtures.py).
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { App } from "../App";
import healthFixture from "./fixtures/health.json";

type Handler = unknown | ((body: unknown) => unknown);
export interface Reply {
  status: number;
  body: unknown;
}
export const reply = (status: number, body: unknown): Reply => ({ status, body });

/** Answers in order; the last answer repeats (e.g. "running", then "success"). */
export class Sequence {
  readonly items: Handler[];
  constructor(items: Handler[]) {
    this.items = items;
  }
}
export const seq = (...items: Handler[]) => new Sequence(items);

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

/**
 * Routes are "METHOD /path" (path without the /api prefix and query string).
 * A value can be: plain data (200, arrays included), a Reply, a function of
 * the request body, or seq(...) for answers in order.
 */
export function mockApi(routes: Record<string, Handler>) {
  const calls: Call[] = [];
  const counters = new Map<string, number>();
  const all: Record<string, Handler> = { "GET /health": healthFixture, ...routes };

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    const path = url.pathname.replace(/^\/api/, "");
    const method = (init?.method ?? "GET").toUpperCase();
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path: `${path}${url.search}`, body });
    const key = `${method} ${path}`;
    if (!(key in all)) {
      return new Response(JSON.stringify({ error: { code: "test_unmocked", message: key } }), { status: 599 });
    }
    let handler = all[key];
    if (handler instanceof Sequence) {
      const n = counters.get(key) ?? 0;
      counters.set(key, n + 1);
      handler = handler.items[Math.min(n, handler.items.length - 1)];
    }
    const value = typeof handler === "function" ? (handler as (b: unknown) => unknown)(body) : handler;
    const { status, body: payload } =
      value && typeof value === "object" && "status" in value && "body" in value
        ? (value as Reply)
        : { status: 200, body: value };
    return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}
