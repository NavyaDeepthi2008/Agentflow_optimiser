// Small typed client for the AgentFlow API. Every failure becomes an ApiError
// with a code and a message the UI can show; nothing is swallowed.
import type {
  CacheCleared,
  Dataset,
  EvaluationConfig,
  EvaluationCreated,
  EvaluationOut,
  EvaluationReport,
  EvaluationRun,
  Health,
  MineRequest,
  MiningOut,
  MiningReport,
  RuleDetail,
  RuleOut,
  ValidationConfig,
  ValidationCreated,
  ValidationOut,
  ValidationResult,
  RunCreated,
  RunDetails,
  RunOut,
  RunRequest,
  TraceEvent,
  TracePage,
  Tool,
} from "./types";

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: { loc: string[]; msg: string }[];

  constructor(status: number, code: string, message: string, details: ApiError["details"] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  get isNetwork(): boolean {
    return this.status === 0;
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string; details?: { loc: string[]; msg: string }[] };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new ApiError(
      0,
      "network_error",
      "The backend is not reachable. Start it with: uvicorn app.api.main:create_app --factory",
    );
  }
  let body: unknown = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!response.ok) {
    const err = (body as ErrorBody | null)?.error;
    throw new ApiError(
      response.status,
      err?.code ?? `http_${response.status}`,
      err?.message ?? `The server answered ${response.status}.`,
      err?.details ?? [],
    );
  }
  return body as T;
}

const post = <T>(path: string, data: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(data) });

export const api = {
  health: () => request<Health>("/health"),
  tools: async () => (await request<{ tools: Tool[] }>("/tools")).tools,
  dataset: (name = "core", version = "1") => request<Dataset>(`/datasets/${name}/${version}`),

  createRun: (body: RunRequest) => post<RunCreated>("/runs", body),
  listRuns: (params: { session_id?: string; status?: string; limit?: number } = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    const suffix = query.toString() ? `?${query}` : "";
    return request<RunOut[]>(`/runs${suffix}`);
  },
  run: (id: string) => request<RunOut>(`/runs/${id}`),
  runDetails: (id: string) => request<RunDetails>(`/runs/${id}/details`),
  /** Every trace event of a run, following the server's paging. */
  fullTrace: async (id: string): Promise<TraceEvent[]> => {
    const events: TraceEvent[] = [];
    let after = 0;
    for (let page = 0; page < 50; page++) {
      const result = await request<TracePage>(`/runs/${id}/trace?after_id=${after}&limit=500`);
      events.push(...result.events);
      if (result.next_after_id == null) break;
      after = result.next_after_id;
    }
    return events;
  },
  clearCache: (sessionId: string) =>
    request<CacheCleared>(`/sessions/${sessionId}/cache`, { method: "DELETE" }),

  createEvaluation: (config: EvaluationConfig) => post<EvaluationCreated>("/evaluations", config),
  listEvaluations: () => request<EvaluationOut[]>("/evaluations"),
  evaluation: async (id: string) => {
    const e = await request<EvaluationOut>(`/evaluations/${id}`);
    return { ...e, report: e.report as unknown as EvaluationReport | null };
  },
  evaluationRuns: (id: string) => request<EvaluationRun[]>(`/evaluations/${id}/runs`),

  // Learning engine (Stage 7)
  mine: async (body: MineRequest) => asMining(await post<MiningOut>("/learning/minings", body)),
  minings: async () => (await request<MiningOut[]>("/learning/minings?limit=5")).map(asMining),
  rules: (status?: string) =>
    request<RuleOut[]>(`/learning/rules${status ? `?status=${encodeURIComponent(status)}` : ""}`),
  rule: (id: string) => request<RuleDetail>(`/learning/rules/${id}`),
  setShadow: (id: string, enabled: boolean) =>
    post<RuleOut>(`/learning/rules/${id}/shadow`, { enabled }),
  validateRule: (id: string, config: ValidationConfig) =>
    post<ValidationCreated>(`/learning/rules/${id}/validations`, config),
  validation: async (id: string) => {
    const v = await request<ValidationOut>(`/learning/validations/${id}`);
    return { ...v, result: v.result as unknown as ValidationResult | null };
  },
  approveRule: (id: string, body: { approved_by: string; note?: string; confirm: boolean }) =>
    post<RuleOut>(`/learning/rules/${id}/approve`, body),
  disableRule: (id: string, body: { reason: string; disabled_by?: string }) =>
    post<RuleOut>(`/learning/rules/${id}/disable`, body),
};

function asMining(m: MiningOut) {
  return { ...m, report: m.report as unknown as MiningReport };
}
