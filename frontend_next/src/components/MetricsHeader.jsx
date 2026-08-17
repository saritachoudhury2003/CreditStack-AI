"use client";

export default function MetricsHeader({ metrics }) {
  // Parse recommendation string e.g. "Pipeline B (Stacking Ensemble)"
  const recRaw = metrics?.recommendation ?? "";
  const recMatch = recRaw.match(/^(.+?)\s*\((.+)\)$/);
  const recName = recMatch ? recMatch[1] : recRaw;
  const recType = recMatch ? recMatch[2] : null;

  const bestAuc = metrics ? metrics.pipeline_b.roc_auc.toFixed(4) : null;
  const bestPrecision = metrics ? metrics.pipeline_b.precision.toFixed(4) : null;

  const aucImprovement =
    metrics
      ? (((metrics.pipeline_b.roc_auc - metrics.pipeline_a.roc_auc) / metrics.pipeline_a.roc_auc) * 100).toFixed(2)
      : null;

  const Skeleton = () => (
    <span className="inline-block h-7 w-24 rounded-md bg-white/10 animate-pulse my-1" />
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      {/* Recommended Model */}
      <div className="glass p-5 flex flex-col justify-between">
        <span className="text-xs text-gray-400 font-medium">Recommended Model</span>
        {metrics ? (
          <>
            <span className="text-2xl font-extrabold text-purple-400 my-1">{recName}</span>
            <span className="text-xs text-emerald-400">{recType ?? "—"}</span>
          </>
        ) : (
          <>
            <Skeleton />
            <span className="text-xs text-gray-500">Loading…</span>
          </>
        )}
      </div>

      {/* Best ROC-AUC */}
      <div className="glass p-5 flex flex-col justify-between">
        <span className="text-xs text-gray-400 font-medium">Best ROC-AUC</span>
        {metrics ? (
          <>
            <span className="text-2xl font-extrabold text-white my-1">{bestAuc}</span>
            <span className="text-xs text-emerald-400">
              {aucImprovement > 0 ? `+${aucImprovement}% vs Baseline` : `${aucImprovement}% vs Baseline`}
            </span>
          </>
        ) : (
          <>
            <Skeleton />
            <span className="text-xs text-gray-500">Loading…</span>
          </>
        )}
      </div>

      {/* Top Precision */}
      <div className="glass p-5 flex flex-col justify-between">
        <span className="text-xs text-gray-400 font-medium">Top Precision</span>
        {metrics ? (
          <>
            <span className="text-2xl font-extrabold text-white my-1">{bestPrecision}</span>
            <span className="text-xs text-emerald-400">Low default risk</span>
          </>
        ) : (
          <>
            <Skeleton />
            <span className="text-xs text-gray-500">Loading…</span>
          </>
        )}
      </div>

      {/* Active Agent LLM */}
      <div className="glass p-5 flex flex-col justify-between">
        <span className="text-xs text-gray-400 font-medium">Active Agent LLM</span>
        <span className="text-xl font-bold text-cyan-400 my-1">llama3.2:1b</span>
        <span className="text-xs text-gray-400">Local Ollama Engine</span>
      </div>
    </div>
  );
}
