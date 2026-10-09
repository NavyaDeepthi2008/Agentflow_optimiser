import { describe, expect, it, vi } from "vitest";
import { mockApi, reply, seq } from "../test/mockApi";
import { ApiError, api } from "./client";

describe("api client", () => {
  it("returns parsed JSON on success", async () => {
    mockApi({ "GET /runs": [] });
    await expect(api.listRuns({ status: "success", limit: 5 })).resolves.toEqual([]);
  });

  it("turns the backend error envelope into an ApiError", async () => {
    mockApi({
      "POST /runs": reply(422, {
        error: { code: "invalid_request", message: "invalid request",
                 details: [{ loc: ["body", "task"], msg: "too long", type: "x" }] },
      }),
    });
    const err = await api.createRun({ task: "x", llm: "gemini" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.code).toBe("invalid_request");
    expect(err.details[0].loc).toEqual(["body", "task"]);
  });

  it("reports a 503 configuration error with the server's message", async () => {
    mockApi({ "POST /runs": reply(503, { error: { code: "llm_not_configured", message: "GEMINI_API_KEY is not set." } }) });
    const err = await api.createRun({ task: "x", llm: "gemini" }).catch((e) => e);
    expect(err.code).toBe("llm_not_configured");
    expect(err.message).toBe("GEMINI_API_KEY is not set.");
  });

  it("reports an unreachable backend as a network error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await api.health().catch((e) => e);
    expect(err.isNetwork).toBe(true);
    expect(err.message).toMatch(/not reachable/);
  });

  it("handles non-JSON error bodies", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("Bad gateway", { status: 502 })));
    const err = await api.health().catch((e) => e);
    expect(err.status).toBe(502);
    expect(err.code).toBe("http_502");
  });

  it("follows trace paging until the server says there is no more", async () => {
    const { calls } = mockApi({
      "GET /runs/abc/trace": seq(
        { run_id: "abc", events: [{ id: 1, type: "run.started", at: "", call_id: null, data: {} }], next_after_id: 1 },
        { run_id: "abc", events: [{ id: 2, type: "run.completed", at: "", call_id: null, data: {} }], next_after_id: null },
      ),
    });
    const events = await api.fullTrace("abc");
    expect(events.map((e) => e.id)).toEqual([1, 2]);
    expect(calls.filter((c) => c.path.startsWith("/runs/abc/trace")).map((c) => c.path)).toEqual([
      "/runs/abc/trace?after_id=0&limit=500",
      "/runs/abc/trace?after_id=1&limit=500",
    ]);
  });
});
