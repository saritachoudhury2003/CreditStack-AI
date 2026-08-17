"use client";

import { useEffect, useState } from "react";
import MetricsHeader from "@/components/MetricsHeader";
import ComparisonChart from "@/components/ComparisonChart";
import CandidateInspector from "@/components/CandidateInspector";
import AgentSwarmStream from "@/components/AgentSwarmStream";
import CsvUploader from "@/components/CsvUploader";

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchMetrics = () => {
    fetch("http://localhost:8000/api/metrics")
      .then((res) => res.json())
      .then((data) => setMetrics(data))
      .catch((err) => console.error("Error fetching metrics:", err));
  };

  useEffect(() => {
    fetchMetrics();
  }, [refreshKey]);

  const handleUploadSuccess = () => {
    setSelectedIdx(0);
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="max-w-[1320px] mx-auto p-4 md:p-6">
      {/* Header Bar */}
      <header className="glass p-4 md:px-6 rounded-2xl flex justify-between items-center mb-6">
        <div className="flex items-center gap-2 text-xl font-extrabold tracking-tight">
          <span className="text-2xl">⚡</span>
          <span>
            CreditStack <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">AI</span>
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold px-3 py-1 rounded-full bg-purple-500/15 text-purple-400 border border-purple-500/30">
            Next.js App Router v15
          </span>
          <span className="text-xs font-bold px-3 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            ● FastAPI Connected (8000)
          </span>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <main>
        {/* CSV File Uploader */}
        <CsvUploader onUploadSuccess={handleUploadSuccess} />

        {/* KPI Stat Cards */}
        <MetricsHeader metrics={metrics} />

        {/* Model Comparison & SHAP Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ComparisonChart metrics={metrics} />
          <CandidateInspector
            key={refreshKey}
            selectedIdx={selectedIdx}
            onSelectCandidate={setSelectedIdx}
          />
        </div>

        {/* Real-time Agent Swarm SSE Stream */}
        <AgentSwarmStream selectedIdx={selectedIdx} />


      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-xs text-gray-500">
        CreditStack AI Next.js App — Built with Next.js App Router, Tailwind CSS, FastAPI & Ollama
      </footer>
    </div>
  );
}
