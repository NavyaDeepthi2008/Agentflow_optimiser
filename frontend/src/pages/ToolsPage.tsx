import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Tool } from "../api/types";
import { Empty, ErrorNotice, Loading } from "../components/ui";
import { useResource } from "../lib/useResource";

const EFFECT: Record<string, string> = {
  read_only: "Reads only",
  side_effect: "Changes data",
  unknown: "Unclassified",
};

function LearnedCell({ tool }: { tool: Tool }) {
  const rules = tool.learned_rules ?? [];
  if (rules.length === 0) {
    return <>{tool.learning_eligible ? "None active" : <span className="hint">Never learned</span>}</>;
  }
  return (
    <>
      {rules.map((r) => (
        <div key={r.rule_id}>
          <Link className="learned-tag" to={`/learning/rules/${r.rule_id}`}>
            {r.kind === "ttl"
              ? `Reuse for ${tool.effective_ttl_seconds ?? "?"} s`
              : `Shared within ${tool.effective_cache_scope ?? "?"}`}
          </Link>
        </div>
      ))}
    </>
  );
}

function reuseRule(freshness: string, ttl: number | null, effect: string): string {
  if (effect !== "read_only") return "Never reused: always executes";
  if (freshness === "live") return "Never reused: always fresh";
  if (freshness === "static") return "Reusable indefinitely (result never changes)";
  return `Reusable for ${ttl ?? "?"} s, then refreshed`;
}

export function ToolsPage() {
  const tools = useResource(() => api.tools(), []);
  return (
    <>
      <header className="page-head">
        <h1>Tools</h1>
        <p>
          The mock tools the agent can call, and the fixed rules the gateway applies to each. Only
          read-only tools can ever reuse a stored result; tools that change data always execute and need
          the task's permission. The reuse rule and sharing columns show each tool's own setting; an
          approved learned rule may lengthen reuse (up to the tool's limit) or widen sharing to a session.
        </p>
      </header>
      {tools.error ? (
        <ErrorNotice error={tools.error} onRetry={tools.reload} />
      ) : !tools.data ? (
        <Loading what="tools" />
      ) : tools.data.length === 0 ? (
        <Empty title="No tools registered" />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Tool</th><th>What it does</th><th>Effect</th><th>Reuse rule</th><th>Shared within</th><th>Permission</th><th className="num">Timeout</th><th>Learned rule (active)</th></tr>
            </thead>
            <tbody>
              {tools.data.map((t) => (
                <tr key={t.name}>
                  <td><strong>{t.name}</strong></td>
                  <td>{t.description}</td>
                  <td>{t.effect === "read_only" ? EFFECT[t.effect] : <span className="tag effect">{EFFECT[t.effect] ?? t.effect}</span>}</td>
                  <td>{reuseRule(t.freshness, t.ttl_seconds ?? null, t.effect)}</td>
                  <td>{t.effect === "read_only" && t.freshness !== "live" ? t.cache_scope : "—"}</td>
                  <td>{t.requires_permission ? "Task must allow it" : "Not needed"}</td>
                  <td className="num">{t.timeout_seconds} s</td>
                  <td><LearnedCell tool={t} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
