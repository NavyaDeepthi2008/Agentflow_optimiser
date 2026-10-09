import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface Resource<T> {
  /** Latest data. While new data for changed inputs loads, the previous value is kept. */
  data: T | undefined;
  error: ApiError | undefined;
  /** True until the first result (or error) for the current inputs arrives. */
  loading: boolean;
  reload: () => void;
}

interface Result<T> {
  key: string;
  data?: T;
  error?: ApiError;
}

/**
 * Load data, and optionally keep polling while `pollWhile(data)` is true.
 * Polling stops on unmount, when the condition becomes false, or on error.
 * State is only set from async callbacks, never synchronously in the effect.
 */
export function useResource<T>(
  load: () => Promise<T>,
  deps: unknown[],
  options: { pollMs?: number; pollWhile?: (data: T) => boolean } = {},
): Resource<T> {
  const [nonce, setNonce] = useState(0);
  const key = JSON.stringify([...deps, nonce]);
  const [result, setResult] = useState<Result<T>>({ key: "" });
  const loadRef = useRef(load);
  const optionsRef = useRef(options);
  useEffect(() => {
    loadRef.current = load;
    optionsRef.current = options;
  });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const value = await loadRef.current();
        if (cancelled) return;
        setResult({ key, data: value });
        const { pollMs, pollWhile } = optionsRef.current;
        if (pollMs && pollWhile?.(value)) timer = setTimeout(tick, pollMs);
      } catch (err) {
        if (cancelled) return;
        const error = err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err));
        setResult((prev) => ({ key, data: prev.data, error }));
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [key]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const current = result.key === key;
  return {
    data: result.data,
    error: current ? result.error : undefined,
    loading: !current,
    reload,
  };
}
