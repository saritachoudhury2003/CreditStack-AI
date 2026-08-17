"use client";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export default function ComparisonChart({ metrics }) {
  const metricsA = metrics ? metrics.pipeline_a : { accuracy: 0.7967, roc_auc: 0.8302, precision: 0.8659, recall: 0.8353, f1: 0.8503 };
  const metricsB = metrics ? metrics.pipeline_b : { accuracy: 0.7805, roc_auc: 0.8539, precision: 0.8919, recall: 0.7765, f1: 0.8302 };

  const data = {
    labels: ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1-Score"],
    datasets: [
      {
        label: "Pipeline A (XGBoost)",
        data: [metricsA.accuracy, metricsA.roc_auc, metricsA.precision, metricsA.recall, metricsA.f1],
        backgroundColor: "rgba(139, 92, 246, 0.65)",
        borderColor: "rgba(139, 92, 246, 1)",
        borderWidth: 1,
        borderRadius: 6,
      },
      {
        label: "Pipeline B (Stacking Ensemble)",
        data: [metricsB.accuracy, metricsB.roc_auc, metricsB.precision, metricsB.recall, metricsB.f1],
        backgroundColor: "rgba(6, 182, 212, 0.65)",
        borderColor: "rgba(6, 182, 212, 1)",
        borderWidth: 1,
        borderRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0.5,
        max: 1.0,
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#9ca3af" },
      },
      x: {
        grid: { display: false },
        ticks: { color: "#9ca3af" },
      },
    },
    plugins: {
      legend: {
        labels: { color: "#f3f4f6", font: { family: "Inter" } },
      },
    },
  };

  return (
    <div className="glass p-6 h-[340px] flex flex-col justify-between">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <span>📊</span> Model Performance Comparison
        </h2>
        <span className="text-xs text-gray-400">Baseline vs Stacking</span>
      </div>
      <div className="h-[250px] w-full">
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}
