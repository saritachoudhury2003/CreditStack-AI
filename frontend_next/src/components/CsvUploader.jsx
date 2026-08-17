"use client";

import { useState } from "react";

export default function CsvUploader({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatusMsg(null);
      setErrorMsg(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setStatusMsg(null);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/upload-csv", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to upload CSV");
      }

      setStatusMsg(data.message);
      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } catch (err) {
      console.error("CSV Upload error:", err);
      setErrorMsg(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="glass p-5 mb-6 flex flex-col md:flex-row items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <span className="text-2xl">📁</span>
        <div>
          <h3 className="text-sm font-bold text-white">Upload Custom Loan CSV Dataset</h3>
          <p className="text-xs text-gray-400">
            Upload your own dataset to retrain models and evaluate candidates dynamically.
          </p>
        </div>
      </div>

      <form onSubmit={handleUpload} className="flex items-center gap-3 w-full md:w-auto">
        <label className="flex-1 md:flex-initial bg-[#151d30] border border-white/10 hover:border-purple-500 text-xs text-gray-300 px-3 py-2 rounded-lg cursor-pointer transition-all truncate">
          <span>{file ? file.name : "Choose .csv file..."}</span>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        <button
          type="submit"
          disabled={!file || isUploading}
          className="bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white font-semibold text-xs px-4 py-2 rounded-lg cursor-pointer transition-all shrink-0"
        >
          {isUploading ? "Uploading & Retraining..." : "Upload & Score ⚡"}
        </button>
      </form>

      {statusMsg && (
        <span className="text-xs text-emerald-400 font-medium">✓ {statusMsg}</span>
      )}
      {errorMsg && (
        <span className="text-xs text-rose-400 font-medium">✗ {errorMsg}</span>
      )}
    </div>
  );
}
