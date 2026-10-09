import { api } from "../api/client";
import type { RunOut } from "../api/types";
import { isActive } from "../components/ui";
import { useResource } from "./useResource";

export const POLL_MS = 800;

/** A run plus its details and trace, refreshed while the run is still active. */
export function useRun(runId: string) {
  const run = useResource(() => api.run(runId), [runId], {
    pollMs: POLL_MS,
    pollWhile: (r: RunOut) => isActive(r.status),
  });
  const status = run.data?.status;
  const details = useResource(() => api.runDetails(runId), [runId, status]);
  const trace = useResource(() => api.fullTrace(runId), [runId, status], {
    pollMs: POLL_MS,
    pollWhile: () => status !== undefined && isActive(status),
  });
  return { run, details, trace };
}
