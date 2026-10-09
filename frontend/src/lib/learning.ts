// Plain-words descriptions of learned rules. Unknown values show "—".
import type { RuleOut, Setting } from "../api/types";
import { MISSING } from "./format";

const SCOPE: Record<string, string> = {
  run: "one run",
  session: "one session",
  global: "everyone",
};

export function describeSetting(kind: string, setting: Setting | null | undefined): string {
  if (!setting) return MISSING;
  if (kind === "ttl") {
    const ttl = setting.ttl_seconds;
    return typeof ttl === "number" ? `reuse for ${fmtSeconds(ttl)}` : MISSING;
  }
  const scope = setting.cache_scope;
  return scope ? `share within ${SCOPE[scope] ?? scope}` : MISSING;
}

export function kindName(kind: string): string {
  return kind === "ttl" ? "Reuse time (TTL)" : kind === "cache_scope" ? "Sharing scope" : kind;
}

export function fmtSeconds(seconds: number): string {
  if (seconds >= 120 && seconds % 60 === 0) return `${seconds / 60} min`;
  return `${Math.round(seconds)} s`;
}

export function ruleTitle(rule: Pick<RuleOut, "tool_name" | "kind" | "current_setting" | "proposed_setting">): string {
  return `${rule.tool_name}: ${describeSetting(rule.kind, rule.current_setting as Setting)} → ${describeSetting(
    rule.kind,
    rule.proposed_setting as Setting,
  )}`;
}

/** What a person can do with a rule in this status (mirrors the backend rules). */
export const CAN = {
  shadowOn: (s: string) => ["proposed", "validated", "rejected", "disabled"].includes(s),
  shadowOff: (s: string) => s === "shadow",
  validate: (s: string) => ["proposed", "shadow", "rejected", "validated", "disabled"].includes(s),
  approve: (s: string) => s === "validated",
  disable: (s: string) => ["proposed", "shadow", "validated", "rejected", "active"].includes(s),
};
