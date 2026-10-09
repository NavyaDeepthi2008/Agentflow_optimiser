// Formatting helpers. The rule: a value the backend did not report is shown as
// "—" (or an explicit phrase), never as 0.

export const MISSING = "—";

type Maybe = number | null | undefined;

export function isKnown(value: Maybe): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function fmtInt(value: Maybe): string {
  return isKnown(value) ? Math.round(value).toLocaleString("en-US") : MISSING;
}

export function fmtMs(value: Maybe): string {
  if (!isKnown(value)) return MISSING;
  if (value >= 10_000) return `${(value / 1000).toFixed(1)} s`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  if (value >= 10) return `${value.toFixed(0)} ms`;
  return `${value.toFixed(2)} ms`;
}

export function fmtPct(value: Maybe, digits = 0): string {
  return isKnown(value) ? `${(value * 100).toFixed(digits)}%` : MISSING;
}

/** A change already expressed in percent (e.g. -27.5). */
export function fmtChange(value: Maybe): string {
  if (!isKnown(value)) return MISSING;
  const rounded = Math.abs(value) < 0.05 ? 0 : value;
  const sign = rounded > 0 ? "+" : rounded < 0 ? "−" : "±";
  return `${sign}${Math.abs(rounded).toFixed(1)}%`;
}

export function fmtCost(value: Maybe, currency = "USD"): string {
  if (!isKnown(value)) return MISSING;
  const digits = value < 0.01 ? 6 : 4;
  return `${value.toFixed(digits)} ${currency}`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return MISSING;
  return date.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : MISSING;
}

export function truncate(text: string, max = 90): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
