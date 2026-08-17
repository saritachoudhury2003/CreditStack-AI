"use client";

import { useState } from "react";

export default function AgentSwarmStream({ selectedIdx }) {
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [analystState, setAnalystState] = useState({ status: "Idle", content: null });
  const [complianceState, setComplianceState] = useState({ status: "Idle", content: null });
  const [communicationState, setCommunicationState] = useState({ status: "Idle", content: null });

  const runSwarm = () => {
    setIsRunning(true);
    setErrorMsg(null);
    setAnalystState({ status: "Waiting...", content: null });
    setComplianceState({ status: "Waiting...", content: null });
    setCommunicationState({ status: "Waiting...", content: null });

    const evtSource = new EventSource(`http://localhost:8000/api/agents/orchestrate-stream/${selectedIdx}`);
    let completed = false;

    evtSource.addEventListener("step", (e) => {
      const data = JSON.parse(e.data);
      const agent = data.agent;

      if (agent === "risk_analyst") {
        if (data.status === "running") {
          setAnalystState({ status: "Running...", content: data.message });
        } else if (data.status === "completed") {
          setAnalystState({ status: "Completed", content: data.result });
        }
      } else if (agent === "compliance") {
        if (data.status === "running") {
          setComplianceState({ status: "Running...", content: data.message });
        } else if (data.status === "completed") {
          setComplianceState({
            status: data.compliance_status,
            content: `Status: ${data.compliance_status}\nNotes: ${data.compliance_notes}`,
          });
        }
      } else if (agent === "communication") {
        if (data.status === "running") {
          setCommunicationState({ status: "Running...", content: data.message });
        } else if (data.status === "completed") {
          setCommunicationState({ status: "Completed", content: data.final_letter });
        } else if (data.status === "blocked") {
          setCommunicationState({ status: "Blocked", content: data.final_letter });
        }
      }
    });

    evtSource.addEventListener("complete", () => {
      completed = true;
      evtSource.close();
      setIsRunning(false);
    });

    // Listen for backend-emitted error events
    evtSource.addEventListener("error", (e) => {
      if (e.data) {
        try {
          const err = JSON.parse(e.data);
          setErrorMsg(err.message || "Unknown error from agent pipeline.");
        } catch {
          setErrorMsg("Agent pipeline encountered an error.");
        }
      }
      completed = true;
      evtSource.close();
      setIsRunning(false);
    });

    // onerror fires on natural stream end in some browsers — only treat as error if not completed
    evtSource.onerror = () => {
      if (!completed) {
        setErrorMsg("SSE connection lost. Is the FastAPI server running on port 8000?");
        evtSource.close();
        setIsRunning(false);
      }
    };
  };

  return (
    <section className="glass p-6 my-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <span>🤖</span> AI Multi-Agent Orchestrator (Next.js SSE Stream)
          </h2>
          <p className="text-xs text-gray-400 mt-1">
            Real-time step-by-step collaboration between Risk Analyst, Compliance Officer, and Communication Agent.
          </p>
        </div>

        <button
          onClick={runSwarm}
          disabled={isRunning}
          className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:opacity-50 text-white font-semibold text-sm px-5 py-2.5 rounded-xl shadow-lg shadow-purple-900/30 transition-all cursor-pointer"
        >
          {isRunning ? "⏳ Agents Running..." : "🚀 Run Agent Swarm"}
        </button>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="mb-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center gap-2">
          <span>⚠️</span> {errorMsg}
        </div>
      )}

      {/* Grid of Agent Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Analyst Card */}
        <div className="glass-sub p-4 flex flex-col justify-between min-h-[160px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-semibold text-sm text-white flex items-center gap-2">
              <span>🕵️</span> 1. Risk Analyst
            </h3>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                analystState.status === "Running..."
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/50 animate-pulse-soft"
                  : analystState.status === "Completed"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                  : "bg-white/10 text-gray-400"
              }`}
            >
              {analystState.status}
            </span>
          </div>
          <div className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
            {analystState.content || (
              <span className="text-gray-500 italic">Click "Run Agent Swarm" to generate risk summary...</span>
            )}
          </div>
        </div>

        {/* Compliance Card */}
        <div className="glass-sub p-4 flex flex-col justify-between min-h-[160px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-semibold text-sm text-white flex items-center gap-2">
              <span>⚖️</span> 2. Compliance Officer
            </h3>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                complianceState.status === "Running..."
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/50 animate-pulse-soft"
                  : complianceState.status === "APPROVED"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                  : complianceState.status === "FLAGGED"
                  ? "bg-rose-500/20 text-rose-400 border border-rose-500/50"
                  : "bg-white/10 text-gray-400"
              }`}
            >
              {complianceState.status}
            </span>
          </div>
          <div className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
            {complianceState.content || (
              <span className="text-gray-500 italic">Audits summary for fair lending compliance...</span>
            )}
          </div>
        </div>

        {/* Communication Card */}
        <div className="glass-sub p-4 flex flex-col justify-between min-h-[160px]">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-semibold text-sm text-white flex items-center gap-2">
              <span>✉️</span> 3. Communication Agent
            </h3>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                communicationState.status === "Running..."
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/50 animate-pulse-soft"
                  : communicationState.status === "Completed"
                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                  : communicationState.status === "Blocked"
                  ? "bg-rose-500/20 text-rose-400 border border-rose-500/50"
                  : "bg-white/10 text-gray-400"
              }`}
            >
              {communicationState.status}
            </span>
          </div>
          <div className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
            {communicationState.content || (
              <span className="text-gray-500 italic">Drafts formal applicant letter once approved...</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
