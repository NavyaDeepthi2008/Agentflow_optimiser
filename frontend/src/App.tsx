import { createContext, useContext } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { NotFound } from "./components/ui";
import { useResource } from "./lib/useResource";
import { ComparePage } from "./pages/ComparePage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { EvaluationsPage } from "./pages/EvaluationsPage";
import { HistoryPage } from "./pages/HistoryPage";
import { LaunchPage } from "./pages/LaunchPage";
import { LearningPage } from "./pages/LearningPage";
import { RulePage } from "./pages/RulePage";
import { RunPage } from "./pages/RunPage";
import { ToolsPage } from "./pages/ToolsPage";

interface BackendState {
  health: Health | undefined;
  reachable: boolean | undefined; // undefined = still checking
}

const BackendContext = createContext<BackendState>({ health: undefined, reachable: undefined });
export const useBackend = () => useContext(BackendContext);

export function App() {
  const health = useResource(() => api.health(), [], { pollMs: 15_000, pollWhile: () => true });
  const state: BackendState = {
    health: health.data,
    reachable: health.error ? false : health.data ? true : undefined,
  };

  return (
    <BackendContext.Provider value={state}>
      <div className="shell">
        <nav className="nav" aria-label="Main">
          <NavLink to="/" className="brand" end>
            <strong>AgentFlow</strong>
            <span>Tool-call optimizer</span>
          </NavLink>
          <ul>
            <li><NavLink to="/" end>Run a task</NavLink></li>
            <li><NavLink to="/runs">Run history</NavLink></li>
            <li><NavLink to="/evaluations">Evaluations</NavLink></li>
            <li><NavLink to="/learning">Learning</NavLink></li>
            <li><NavLink to="/tools">Tools</NavLink></li>
          </ul>
          <div className="nav-foot" aria-live="polite">
            <span>
              <i className={`dot ${state.reachable === false ? "bad" : state.reachable ? "on" : "off"}`} />
              {state.reachable === false ? "Backend unreachable" : state.reachable ? "Backend connected" : "Connecting"}
            </span>
            {state.health && (
              <span>
                <i className={`dot ${state.health.gemini_configured ? "on" : "off"}`} />
                {state.health.gemini_configured
                  ? `Gemini: ${state.health.gemini_model}`
                  : "Gemini not configured"}
              </span>
            )}
          </div>
        </nav>
        <main>
          <Routes>
            <Route path="/" element={<LaunchPage />} />
            <Route path="/runs" element={<HistoryPage />} />
            <Route path="/runs/:runId" element={<RunPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/evaluations" element={<EvaluationsPage />} />
            <Route path="/evaluations/:evaluationId" element={<EvaluationPage />} />
            <Route path="/learning" element={<LearningPage />} />
            <Route path="/learning/rules/:ruleId" element={<RulePage />} />
            <Route path="/tools" element={<ToolsPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
      </div>
    </BackendContext.Provider>
  );
}
