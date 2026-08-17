"use client";

import { useState } from "react";

export default function UnderwriterChat({ selectedIdx }) {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      avatar: "🤖",
      content:
        "Hello! I am your AI Underwriting Assistant. Ask me about the selected applicant's risk factors, verification documents, or simulate custom income/DTI scenarios!",
    },
  ]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userText = query.trim();
    setQuery("");
    setMessages((prev) => [...prev, { role: "user", avatar: "👤", content: userText }]);

    setIsLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/agent/underwriter-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idx: selectedIdx, user_query: userText }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", avatar: "🤖", content: data.agent_response },
      ]);
    } catch (err) {
      console.error("Error sending chat query:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", avatar: "🤖", content: "Error connecting to underwriter agent." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="glass p-6 my-6">
      <div className="mb-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <span>💬</span> Interactive Underwriter Assistant
        </h2>
        <p className="text-xs text-gray-400 mt-1">
          Chat with the Underwriter Agent for Applicant #{selectedIdx}. It can fetch document verification status, explain SHAP drivers, or simulate alternate scenarios live!
        </p>
      </div>

      <div className="space-y-4">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Ask about Applicant #${selectedIdx}... (e.g. Are documents verified? Or simulate income 8000 and DTI 0.15)`}
            className="flex-1 bg-[#151d30] border border-white/10 text-white text-sm rounded-xl px-4 py-2.5 outline-none focus:border-purple-500"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-white/10 hover:bg-white/20 text-white text-sm font-semibold px-5 py-2.5 rounded-xl cursor-pointer transition-all"
          >
            {isLoading ? "Thinking..." : "Send ↵"}
          </button>
        </form>

        {/* Message List */}
        <div className="max-h-[260px] overflow-y-auto space-y-3 p-2 border border-white/5 rounded-xl bg-black/20">
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`flex gap-3 text-xs leading-relaxed max-w-[85%] ${
                m.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
              }`}
            >
              <div className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center shrink-0 text-sm">
                {m.avatar}
              </div>
              <div
                className={`p-3 rounded-xl border ${
                  m.role === "user"
                    ? "bg-purple-600/20 border-purple-500/40 text-purple-100"
                    : "bg-[#1a243c] border-white/10 text-gray-200"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
