"use client";

import { useEffect, useState, useCallback } from "react";

export default function CandidateInspector({ selectedIdx, onSelectCandidate }) {
  const [candidates, setCandidates] = useState([]);
  const [applicantData, setApplicantData] = useState(null);
  const [loadingApplicant, setLoadingApplicant] = useState(false);
  const [error, setError] = useState(null);
  const [backendReady, setBackendReady] = useState(false);

  // Fetch candidate list once backend is confirmed ready
  const fetchCandidates = useCallback(() => {
    fetch("http://localhost:8000/api/candidates")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setCandidates(data.candidates || []);
        setBackendReady(true);
        setError(null);
      })
      .catch((err) => {
        setError("Backend not reachable. Is the FastAPI server running on port 8000?");
        console.error("Error fetching candidates:", err);
      });
  }, []);

  useEffect(() => {
    fetchCandidates();
  }, [fetchCandidates]);

  // Fetch applicant detail only when backend is ready
  useEffect(() => {
    if (!backendReady) return;
    setLoadingApplicant(true);
    setApplicantData(null);
    fetch(`http://localhost:8000/api/applicant/${selectedIdx}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setApplicantData(data);
        setLoadingApplicant(false);
      })
      .catch((err) => {
        console.error("Error fetching applicant detail:", err);
        setLoadingApplicant(false);
      });
  }, [selectedIdx, backendReady]);

  const probPercent = applicantData
    ? (applicantData.probability_pipeline_b * 100).toFixed(1) + "%"
    : "--%";

  const decision = applicantData ? applicantData.decision : "--";

  const badgeColor =
    decision === "APPROVE"
      ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/50"
      : decision === "REVIEW"
      ? "bg-amber-500/20 text-amber-400 border-amber-500/50"
      : decision === "REJECT"
      ? "bg-rose-500/20 text-rose-400 border-rose-500/50"
      : "bg-gray-500/20 text-gray-400 border-gray-500/50";

  if (error) {
    return (
      <div className="glass p-6 h-[340px] flex flex-col items-center justify-center gap-4">
        <span className="text-2xl">⚠️</span>
        <p className="text-sm text-rose-400 text-center">{error}</p>
        <button
          onClick={fetchCandidates}
          className="px-4 py-2 text-xs font-bold rounded-lg bg-purple-600/30 text-purple-300 border border-purple-500/40 hover:bg-purple-600/50 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="glass p-6 h-[340px] flex flex-col justify-between">
      <div>
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span>🎯</span> Candidate Inspector &amp; SHAP
          </h2>
          <span className="text-xs text-gray-400">Test Set Evaluation</span>
        </div>

        {/* Dropdown Selector */}
        <div className="mb-4">
          <label className="text-xs text-gray-400 block mb-1">Select Candidate:</label>
          <select
            value={selectedIdx}
            onChange={(e) => onSelectCandidate(parseInt(e.target.value))}
            className="w-full bg-[#151d30] border border-white/10 text-white text-sm rounded-lg px-3 py-2 outline-none focus:border-purple-500 cursor-pointer"
            disabled={!backendReady || candidates.length === 0}
          >
            {candidates.length === 0 ? (
              <option>Loading candidates…</option>
            ) : (
              candidates.map((c) => (
                <option key={c.index} value={c.index}>
                  {c.id}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Score Box */}
        <div className="flex justify-between items-center p-3 bg-white/[0.03] rounded-lg mb-4">
          <div>
            <span className="text-xs text-gray-400">Approval Odds: </span>
            {loadingApplicant ? (
              <span className="inline-block h-4 w-16 rounded bg-white/10 animate-pulse ml-1 align-middle" />
            ) : (
              <span className="text-lg font-bold text-cyan-400 ml-1">{probPercent}</span>
            )}
          </div>
          {loadingApplicant ? (
            <span className="inline-block h-6 w-20 rounded-full bg-white/10 animate-pulse" />
          ) : (
            <span className={`text-xs font-bold px-3 py-1 rounded-full border uppercase ${badgeColor}`}>
              {decision}
            </span>
          )}
        </div>
      </div>

      {/* Top SHAP Feature Bars */}
      <div>
        <h3 className="text-xs font-semibold text-gray-400 mb-2">Top SHAP Risk Drivers:</h3>
        <div className="space-y-1.5">
          {loadingApplicant ? (
            [1, 2, 3].map((i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="h-3 w-1/3 rounded bg-white/10 animate-pulse" />
                <div className="h-2 w-1/2 rounded-full bg-white/10 animate-pulse" />
                <div className="h-3 w-10 rounded bg-white/10 animate-pulse" />
              </div>
            ))
          ) : applicantData && applicantData.top_features ? (
            applicantData.top_features.slice(0, 3).map((f, i) => {
              const isPos = f.impact > 0;
              const widthPct = Math.min(100, Math.max(15, f.abs_impact * 250));
              return (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="w-1/3 truncate text-gray-300">{f.feature}</span>
                  <div className="w-1/2 bg-white/5 h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${isPos ? "bg-emerald-400" : "bg-rose-400"}`}
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <span className={`font-mono text-[11px] ${isPos ? "text-emerald-400" : "text-rose-400"}`}>
                    {isPos ? "+" : ""}{f.impact.toFixed(3)}
                  </span>
                </div>
              );
            })
          ) : (
            <span className="text-xs text-gray-500 italic">No SHAP data available.</span>
          )}
        </div>
      </div>
    </div>
  );
}
